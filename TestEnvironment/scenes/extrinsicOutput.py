import bpy
import json
import mathutils

output = {}

for obj in bpy.data.objects:
    if obj.type == 'CAMERA':

        name = obj.name

        # Welt-Transform holen
        world_matrix = obj.matrix_world.copy()

        # Translation
        T = world_matrix.to_translation()
        T = [float(T.x), float(T.y), float(T.z)]

        # Rotation als Matrix
        R = world_matrix.to_3x3()
        R = [[float(R[i][j]) for j in range(3)] for i in range(3)]

        output[name] = {
            "R": R,
            "T": T
        }

# JSON speichern
save_path = bpy.path.abspath("//groundtruth_extrinsics.json")
with open(save_path, 'w') as f:
    json.dump(output, f, indent=4)

print("Export complete →", save_path)
