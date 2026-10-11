"""Smooth a rigged model for Roblox without touching its rig, weights, UVs or materials.

Usage:
    python3 smooth_rigged_model.py INPUT.(fbx|glb|gltf) OUTPUT.fbx [--subdiv N] [--max-tris 20000]

What it does to each skinned/static mesh:
  1. Smooth shading, keeping hard edges sharper than 60 degrees (no geometry change).
  2. Optional Catmull-Clark subdivision (applied to the rest pose, so weights and UVs are
     interpolated onto the new vertices). Skipped for any mesh that would exceed the
     triangle budget or that has shape keys (Roblox dynamic heads).
  3. Re-limits skin weights to Roblox's 4 influences per vertex and normalizes them.
It never edits the armature, bone names, vertex group names, materials or UV maps.
"""
import argparse
import math
import sys

import bpy

ROBLOX_MAX_INFLUENCES = 4


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--subdiv", type=int, default=1, help="subdivision levels (0 = shading only)")
    p.add_argument("--max-tris", type=int, default=20000, help="per-mesh triangle budget")
    p.add_argument("--smooth-angle", type=float, default=60.0, help="edges sharper than this stay hard")
    return p.parse_args(argv)


def tri_count(obj):
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)


def snapshot(objs):
    """Everything that must come out of the process unchanged."""
    snap = {"armatures": {}, "meshes": {}}
    for o in objs:
        if o.type == "ARMATURE":
            snap["armatures"][o.name] = [(b.name, b.parent.name if b.parent else None) for b in o.data.bones]
        elif o.type == "MESH":
            snap["meshes"][o.name] = {
                "groups": sorted(g.name for g in o.vertex_groups),
                "materials": [s.material.name if s.material else None for s in o.material_slots],
                "uvs": [uv.name for uv in o.data.uv_layers],
                "armature": next((m.object.name for m in o.modifiers if m.type == "ARMATURE" and m.object), None),
                "parent": o.parent.name if o.parent else None,
            }
    return snap


def import_model(path):
    ext = path.lower().rsplit(".", 1)[-1]
    if ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=path, use_custom_normals=False)
    elif ext in ("glb", "gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == "blend":
        bpy.ops.wm.open_mainfile(filepath=path)
    else:
        sys.exit(f"Unsupported format: {ext}")


def smooth_mesh(obj, args, report):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    before = tri_count(obj)
    has_shape_keys = obj.data.shape_keys is not None

    # Subdivide the rest mesh (modifier first in the stack, applied before the armature).
    levels = args.subdiv
    while levels > 0 and before * (4 ** levels) > args.max_tris:
        levels -= 1
    note = ""
    if has_shape_keys and levels:
        levels, note = 0, "has shape keys, subdivision skipped"
    elif levels < args.subdiv:
        note = f"subdivision reduced to {levels} to stay under {args.max_tris} tris"
    if levels:
        mod = obj.modifiers.new("RobloxSmooth", "SUBSURF")
        mod.levels = mod.render_levels = levels
        mod.uv_smooth = "PRESERVE_BOUNDARIES"  # keeps UV seams from stretching
        mod.boundary_smooth = "ALL"
        while obj.modifiers[0] != mod:
            bpy.ops.object.modifier_move_up(modifier=mod.name)
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # Smooth shading with hard edges kept above the angle.
    bpy.ops.object.shade_smooth_by_angle(angle=math.radians(args.smooth_angle), keep_sharp_edges=True)

    # Roblox: max 4 bone influences per vertex, normalized.
    if obj.vertex_groups:
        bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=ROBLOX_MAX_INFLUENCES)
        bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)

    report.append((obj.name, before, tri_count(obj), levels, note))


def verify(before, after, objs, args):
    problems = []
    if before["armatures"] != after["armatures"]:
        problems.append("armature/bone hierarchy changed")
    for name, info in before["meshes"].items():
        if after["meshes"].get(name) != info:
            problems.append(f"{name}: groups/materials/UVs/armature binding changed")
    for o in objs:
        if o.type != "MESH":
            continue
        worst = max((sum(1 for g in v.groups if g.weight > 0) for v in o.data.vertices), default=0)
        if worst > ROBLOX_MAX_INFLUENCES:
            problems.append(f"{o.name}: {worst} influences on a vertex")
        if tri_count(o) > args.max_tris:
            problems.append(f"{o.name}: {tri_count(o)} tris (over {args.max_tris})")
    return problems


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import_model(args.input)
    objs = list(bpy.context.scene.objects)
    before = snapshot(objs)

    report = []
    for obj in [o for o in objs if o.type == "MESH"]:
        smooth_mesh(obj, args, report)

    problems = verify(before, snapshot(objs), objs, args)

    bpy.ops.export_scene.fbx(
        filepath=args.output,
        use_selection=False,
        object_types={"ARMATURE", "MESH"},
        add_leaf_bones=False,
        bake_anim=True,
        mesh_smooth_type="FACE",
        use_armature_deform_only=False,
        path_mode="COPY",
        embed_textures=True,
    )

    print("\nMesh                           tris before -> after  subdiv  note")
    for name, b, a, lv, note in report:
        print(f"{name:30} {b:>11} -> {a:<6} {lv:>6}  {note}")
    print("\nRig/weights/UV/material check:", "OK" if not problems else "PROBLEMS")
    for p in problems:
        print("  -", p)
    print("Wrote", args.output)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
