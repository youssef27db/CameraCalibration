import bpy
import json
import mathutils

# Blender -> OpenCV Koordinatentransformation
# Blender:  X rechts, Y nach vorne, Z nach oben
# OpenCV:   X rechts, Y nach unten, Z nach vorne
B_TO_CV = mathutils.Matrix((
    (1,  0,  0),
    (0, -1,  0),
    (0,  0, -1),
))

# Mapping: so heißen deine Kamera-Objekte in Blender
CAM_OBJECTS = {
    "Center": "Camera_Center",
    "Up1":    "Camera_Up1",
    "Up2":    "Camera_Up2",
    "Up3":    "Camera_Up3",
    "Down1":  "Camera_Down1",
    "Down2":  "Camera_Down2",
    "Down3":  "Camera_Down3",
    "Left1":  "Camera_Left1",
    "Left2":  "Camera_Left2",
    "Left3":  "Camera_Left3",
    "Right1": "Camera_Right1",
    "Right2": "Camera_Right2",
    "Right3": "Camera_Right3",
}


def get_pose_cv(obj: bpy.types.Object):
    """
    Liefert die Pose der Kamera im OpenCV-System als Welt->Kamera-Extrinsics:

        X_cam = R_cv * X_world + t_cv

    Wichtig: wir benutzen matrix_world.inverted(), damit wir
    die OpenCV-Konvention Welt->Kamera bekommen.
    """
    # Welt->Kamera (OpenGL-/Blender-Style)
    M_wc = obj.matrix_world.inverted()
    loc, rot_quat, scale = M_wc.decompose()

    # reine Rotation ohne Scale
    R_bl = rot_quat.to_matrix().to_3x3()
    t_bl = loc

    # Blender -> OpenCV-Achsen
    R_cv = B_TO_CV @ R_bl @ B_TO_CV.transposed()
    t_cv = B_TO_CV @ t_bl

    return R_cv, t_cv


def export_relative_extrinsics(save_path: str):
    """
    Exportiert für jede Kamera Extrinsics relativ zur Center-Kamera:
        X_cam = R_rel * X_center + T_rel

    R_rel und T_rel sind im OpenCV-System und kompatibel mit deinen
    Kalibrierungs-Extrinsics aus OpenCV.
    """
    # Referenzkamera = Center
    ref_name = "Center"
    ref_obj = bpy.data.objects[CAM_OBJECTS[ref_name]]

    R0, t0 = get_pose_cv(ref_obj)   # Welt->Center
    R0_T = R0.transposed()

    output = {}

    for cam_id, obj_name in CAM_OBJECTS.items():
        cam_obj = bpy.data.objects[obj_name]
        Rj, tj = get_pose_cv(cam_obj)   # Welt->Cam_j

        if cam_id == "Center":
            # Referenz: Identität
            R_rel = mathutils.Matrix.Identity(3)
            T_rel = mathutils.Vector((0.0, 0.0, 0.0))
        else:
            # relative Extrinsics von Center zu Cam_j:
            # R_rel = R_j * R_0^T
            # T_rel = t_j - R_rel * t_0
            R_rel = Rj @ R0_T
            T_rel = tj - R_rel @ t0

        R_list = [[float(R_rel[i][j]) for j in range(3)] for i in range(3)]
        T_list = [float(T_rel.x), float(T_rel.y), float(T_rel.z)]

        output[cam_id] = {
            "rotationMatrix": R_list,
            "translationVector": T_list,
        }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print("Export complete →", save_path)


# Ausführen: speichert im gleichen Ordner wie die .blend-Datei
save_path = bpy.path.abspath("//groundtruth_extrinsics_relative.json")
export_relative_extrinsics(save_path)
