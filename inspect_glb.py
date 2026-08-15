import struct, json, sys

path = "frontend/assets/avatar.glb"

with open(path, "rb") as f:
    f.read(12)  # skip 12-byte GLB header (magic, version, length)
    chunk_len = struct.unpack("<I", f.read(4))[0]
    f.read(4)  # chunk type (should be 'JSON')
    json_data = json.loads(f.read(chunk_len))

meshes = json_data.get("meshes", [])
print(f"Total meshes: {len(meshes)}\n")

for i, mesh in enumerate(meshes):
    name = mesh.get("name")
    print(f"Mesh {i}: name={name!r}")
    for j, prim in enumerate(mesh.get("primitives", [])):
        targets = prim.get("targets", [])
        print(f"  primitive {j}: {len(targets)} morph targets")
    extras = mesh.get("extras", {})
    target_names = extras.get("targetNames")
    if target_names:
        print(f"  targetNames ({len(target_names)}):")
        for n in target_names:
            print(f"    - {n}")
    else:
        print("  targetNames: (none found in extras)")
    print()