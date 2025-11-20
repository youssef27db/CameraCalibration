import bpy
import random
import math

# wie stark "kaputt machen"
max_trans = 0.05      # max. Verschiebung in Blender-Units (~Meter)
max_rot_deg = 2.0     # max. Drehung in Grad

for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        # Translation leicht verrauschen
        obj.location.x += random.uniform(-max_trans, max_trans)
        obj.location.y += random.uniform(-max_trans, max_trans)
        obj.location.z += random.uniform(-max_trans, max_trans)

        # Rotation leicht verrauschen (Euler XYZ)
        e = obj.rotation_euler
        e.x += math.radians(random.uniform(-max_rot_deg, max_rot_deg))
        e.y += math.radians(random.uniform(-max_rot_deg, max_rot_deg))
        e.z += math.radians(random.uniform(-max_rot_deg, max_rot_deg))

print("Cameras jittered (entkalibriert).")
