import bpy
import json
import mathutils

# Blender -> OpenCV Koordinaten (Achsen flip)
B_TO_CV = mathutils.Matrix((
    (1,  0,  0),
    (0, -1,  0),
    (0,  0, -1),
))

# Kamera-Objekte
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

# Name deines Parents (Empty). Falls anders: anpassen.
RIG_PARENT_NAME = "Leer"

def get_uniform_world_scale_from_parent(parent_obj: bpy.types.Object) -> float:
    """
    Erwartet uniform scale (x==y==z). Bei dir: 0.5.
    Gibt world-scale des Parent zurück.
    """
    if parent_obj is None:
        return 1.0
    sx, sy, sz = parent_obj.matrix_world.to_scale()
    # Wenn nicht uniform, warnen (dann wird's komplizierter)
    if abs(sx - sy) > 1e-6 or abs(sx - sz) > 1e-6:
        print(f"[WARN] Parent scale not uniform: {sx}, {sy}, {sz}. Using sx.")
    return float(sx)

def matrix_world_no_scale(obj: bpy.types.Object) -> mathutils.Matrix:
    """
    Entfernt Scale-Anteile aus der matrix_world, behält aber Translation + Rotation.
    """
    loc, rot, _ = obj.matrix_world.decompose()
    M = rot.to_matrix().to_4x4()
    M.translation = loc
    return M

def get_pose_cv_world_to_cam(obj: bpy.types.Object, scale_correction: float):
    """
    Liefert OpenCV-Extrinsics Welt->Kamera:
        X_cam = R * X_world + t

    scale_correction: Faktor, um Parent-Scale rauszurechnen.
      - Wenn Parent scale=0.5, dann scale_correction = 1/0.5 = 2.0
    """
    # Weltpose ohne Scale (damit keine Scher/Scale in der Rotation steckt)
    M_w = matrix_world_no_scale(obj)

    # Welt -> Kamera
    M_wc = M_w.inverted()

    loc, rot, _ = M_wc.decompose()
    R_bl = rot.to_matrix().to_3x3()
    t_bl = loc

    # Blender -> OpenCV Achsen
    R_cv = B_TO_CV @ R_bl @ B_TO_CV.transposed()
    t_cv = B_TO_CV @ t_bl

    # >>> Skalenkorrektur: Translation in "echte" Meter bringen
    t_cv = t_cv * scale_correction

    return R_cv, t_cv

def export_relative_extrinsics(save_path: str):
    parent = bpy.data.objects.get(RIG_PARENT_NAME, None)
    parent_scale = get_uniform_world_scale_from_parent(parent)
    scale_correction = 1.0 / parent_scale if parent_scale != 0 else 1.0

    print(f"[INFO] Parent '{RIG_PARENT_NAME}' world scale = {parent_scale}")
    print(f"[INFO] scale_correction = {scale_correction}")

    ref_obj = bpy.data.objects[CAM_OBJECTS["Center"]]
    R0, t0 = get_pose_cv_world_to_cam(ref_obj, scale_correction)
    R0_T = R0.transposed()

    out = {}

    for cam_id, obj_name in CAM_OBJECTS.items():
        cam_obj = bpy.data.objects[obj_name]
        Rj, tj = get_pose_cv_world_to_cam(cam_obj, scale_correction)

        if cam_id == "Center":
            R_rel = mathutils.Matrix.Identity(3)
            T_rel = mathutils.Vector((0.0, 0.0, 0.0))
        else:
            # relative Extrinsics von Center -> Cam_j (OpenCV kompatibel)
            R_rel = Rj @ R0_T
            T_rel = tj - R_rel @ t0

        out[cam_id] = {
            "rotationMatrix": [[float(R_rel[i][j]) for j in range(3)] for i in range(3)],
            "translationVector": [float(T_rel.x), float(T_rel.y), float(T_rel.z)],
        }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Export complete →", save_path)

# Speichern im .blend-Ordner
save_path = bpy.path.abspath("//groundtruth_extrinsics_Rig1_set5.json")
export_relative_extrinsics(save_path)
