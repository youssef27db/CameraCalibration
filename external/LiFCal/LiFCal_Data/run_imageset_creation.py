import os
import re
import shutil
from collections import defaultdict
import depthmap

"""
@brief Organizes focus images by camera and generates depth maps for LiFCal recalibration.

"""

ROOT = "/data/external/LiFCal/LiFCal_Data/Recalibration/LiFCal_Imageset"
FOCUS_DIR = os.path.join(ROOT, "focus")
DEPTH_DIR = os.path.join(ROOT, "depth")

# If True: original files in focus root directory are deleted after sorting
CLEAN_FOCUS_ROOT_AFTER = True

CAMERA_IDS = [
    "Center", "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3"
]

# Expected filenames: pose000_Center.png ... pose000_Down3.png
PATTERN = re.compile(
    r"^(pose\d+)_(Center|Up[123]|Down[123]|Left[123]|Right[123])\.(png|jpg|jpeg)$",
    re.IGNORECASE
)

def ensure_camera_subfolders(base_dir: str):
    """
    @brief Create subdirectory for each camera ID under base_dir.
    
    @param base_dir Base directory path
    """
    os.makedirs(base_dir, exist_ok=True)
    for cid in CAMERA_IDS:
        os.makedirs(os.path.join(base_dir, cid), exist_ok=True)

def group_focus_images_by_pose():
    """
    @brief Group focus images by pose ID from flat directory structure.
    
    @return Dictionary mapping pose IDs to camera-path dictionaries
    """
    pose_to_cam = defaultdict(dict)
    for fn in sorted(os.listdir(FOCUS_DIR)):
        m = PATTERN.match(fn)
        if not m:
            continue
        pose, cam, ext = m.group(1), m.group(2), m.group(3)
        pose_to_cam[pose][cam] = os.path.join(FOCUS_DIR, fn)
    return pose_to_cam

def copy_focus_into_camera_folders(pose: str, cam_to_path: dict):
    """
    @brief Copy focus images into camera-specific subdirectories.
    
    @param pose Pose identifier string (e.g., "pose000")
    @param cam_to_path Dictionary mapping camera IDs to source image paths
    """
    for cam, src in cam_to_path.items():
        ext = os.path.splitext(src)[1].lower()
        dst = os.path.join(FOCUS_DIR, cam, f"{pose}_{cam}{ext}")
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)

def cleanup_focus_root():
    """
    @brief Delete only loose files directly in focus/ (not in subdirectories).
    """
    for fn in os.listdir(FOCUS_DIR):
        p = os.path.join(FOCUS_DIR, fn)
        if os.path.isfile(p) and PATTERN.match(fn):
            os.remove(p)

def main():
    """
    @brief Main pipeline: organize focus images and generate depth maps.
    """
    if not os.path.isdir(FOCUS_DIR):
        raise SystemExit(f"focus folder not found: {FOCUS_DIR}")

    ensure_camera_subfolders(FOCUS_DIR)
    ensure_camera_subfolders(DEPTH_DIR)

    pose_to_cam = group_focus_images_by_pose()
    if not pose_to_cam:
        raise SystemExit("No focus images found. Expected: pose000_Center.png etc.")

    poses = sorted(pose_to_cam.keys())
    print("Found poses:", len(poses))

    for i, pose in enumerate(poses, start=1):
        cam_to_focus_path = pose_to_cam[pose]

        # Sort focus images into camera folders
        copy_focus_into_camera_folders(pose, cam_to_focus_path)

        # Generate depth maps for this pose
        depthmaps = depthmap.generate_depthmaps_for_pose(cam_to_focus_path)

        # Save depth maps: depth/<CamId>/<pose>_<CamId>_depth.png
        for cam, depth_u16 in depthmaps.items():
            out_path = os.path.join(DEPTH_DIR, cam, f"{pose}_{cam}_depth.png")
            ok = depthmap.imwrite_u16(out_path, depth_u16)
            if not ok:
                print("FAILED write:", out_path)

        print(f"[{i}/{len(poses)}] {pose}: views={len(cam_to_focus_path)} depthmaps={len(depthmaps)}")

    if CLEAN_FOCUS_ROOT_AFTER:
        cleanup_focus_root()

    print("DONE.")
    print("Focus by camera:", FOCUS_DIR)
    print("Depth by camera:", DEPTH_DIR)

if __name__ == "__main__":
    main()
