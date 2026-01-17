import json
import numpy as np
from pathlib import Path

# Paths
THIS_DIR = Path(__file__).resolve().parent
CALIB_LFC_DIR = THIS_DIR.parent          # calibrationLFC/
PROJECT_ROOT = CALIB_LFC_DIR.parent      # CameraCalibration/
RESULTS_DIR = CALIB_LFC_DIR / "results"

# Groundtruth base directory
GT_DIR = PROJECT_ROOT / "TestEnvironment" / "params"

# Camera order
CAM_IDS = [
    "Center",
    "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3",
]

# Calibration JSONs
CALIB_FILES = {
    "imageset1": RESULTS_DIR / "calibration_initial_imageset1_20251215_173701.json",
    "imageset2": RESULTS_DIR / "calibration_initial_imageset2_20251215_181903.json",
    "imageset3": RESULTS_DIR / "calibration_initial_imageset3_20251215_185041.json",
    "imageset4": RESULTS_DIR / "calibration_initial_imageset4_20251215_165644.json",
    "imageset5": RESULTS_DIR / "calibration_initial_imageset5_20251217_011656.json",
    "imageset6": RESULTS_DIR / "calibration_initial_imageset6_20251215_201527.json",
    "imageset7": RESULTS_DIR / "calibration_initial_imageset7_20251215_203849.json",
}

# Groundtruth per calibration
GT_FILES = {
    # Rig0 for imageset1–4
    "imageset1": GT_DIR / "groundtruth_extrinsics_Rig0.json",
    "imageset2": GT_DIR / "groundtruth_extrinsics_Rig0.json",
    "imageset3": GT_DIR / "groundtruth_extrinsics_Rig0.json",
    "imageset4": GT_DIR / "groundtruth_extrinsics_Rig0.json",

    # Different rigs/sets:
    "imageset5": GT_DIR / "groundtruth_extrinsics_Rig1_set5.json",
    "imageset6": GT_DIR / "groundtruth_extrinsics_Rig2_set6.json",
    "imageset7": GT_DIR / "groundtruth_extrinsics_Rig3_set7.json",  
}

# Scenario grouping for thesis table
SCENARIOS = {
    "A_Robustheit": ["imageset1", "imageset2", "imageset3", "imageset4"],
    "B_Generalisation":    ["imageset5", "imageset6", "imageset7"],
}

# Helper functions
def rot_angle_deg(R):
    """Extract rotation angle (degrees) from 3x3 rotation matrix."""
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def as_vec3(v):
    """Convert any translation vector to robust 3-element vector."""
    arr = np.array(v, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected 3 entries for translation, got {arr.size}")
    return arr


def compute_metrics_for_file(calib_path: Path, gt_path: Path):
    """
    Loads a calibration JSON and groundtruth JSON and computes:
      - numPoses
      - mean ΔT (m)
      - mean ΔR (°)
      - mean reprojection error (px, from intrinsics)

    Definitions as in your extrinsic plot:
      ΔR_cam = angle(R_est * R_gt^T)
      ΔT_cam = || T_est - T_gt ||_2
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    with open(calib_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    meta = calib.get("meta", {})
    extr_est = calib["state"]["extrinsics"]
    intr_est = calib["state"]["intrinsics"]

    bundle_adjust = bool(meta.get("bundleAdjust", False))
    num_poses = int(meta.get("numPoses", -1))

    rot_err = []
    trans_err = []
    reproj_err = []

    for cam in CAM_IDS:
        if cam not in extr_est or cam not in gt:
            print(f"[WARN] Camera '{cam}' not in both files for {calib_path.name}, skipping this cam.")
            continue

        # Extrinsics
        R_est = np.array(extr_est[cam]["rotationMatrix"], dtype=float)
        R_gt  = np.array(gt[cam]["rotationMatrix"], dtype=float)

        T_est = as_vec3(extr_est[cam]["translationVector"])
        T_gt  = as_vec3(gt[cam]["translationVector"])

        # Rotation error as in plot
        R_err = R_est @ R_gt.T
        ang = rot_angle_deg(R_err)

        # Translation error (absolute, m)
        d_abs = float(np.linalg.norm(T_est - T_gt))

        rot_err.append(ang)
        trans_err.append(d_abs)

        # Intrinsic reprojection error
        if cam in intr_est:
            reproj_err.append(float(intr_est[cam]["reprojectionError"]))

    # Mean values
    mean_dT = float(np.mean(trans_err))     if trans_err else float("nan")
    mean_dR = float(np.mean(rot_err))       if rot_err else float("nan")
    mean_reproj = float(np.mean(reproj_err)) if reproj_err else float("nan")

    return {
        "bundleAdjust": bundle_adjust,
        "numPoses": num_poses,
        "mean_dT": mean_dT,
        "mean_dR": mean_dR,
        "mean_reproj": mean_reproj,
    }

# Table output (console)
def print_per_file_table(results_by_name):
    """
    Prints one line per calibration JSON.
    """
    header = (
        f"{'Name':<12} "
        f"{'#Poses':>6} "
        f"{'BA':<4} "
        f"{'mean dT [m]':>12} "
        f"{'mean dR [deg]':>13} "
        f"{'mean reproj [px]':>16}"
    )
    print(header)
    print("-" * len(header))

    for name, res in sorted(results_by_name.items()):
        ba_flag = "yes" if res["bundleAdjust"] else "no"
        print(
            f"{name:<12} "
            f"{res['numPoses']:6d} "
            f"{ba_flag:<4} "
            f"{res['mean_dT']:12.5f} "
            f"{res['mean_dR']:12.5f} "
            f"{res['mean_reproj']:16.5f}"
        )

def print_scenario_table(results_by_name):
    """
    Aggregates over scenarios (Robustness / Generalization)
    and prints a compact table.
    """
    print("\n\nScenario Summary:\n")

    header = (
        f"{'Szenario':<22} "
        f"{'#Runs':>6} "
        f"{'Avg#Poses':>9} "
        f"{'mean dT [m]':>12} "
        f"{'mean dR [deg]':>13} "
        f"{'mean reproj [px]':>16}"
    )
    print(header)
    print("-" * len(header))

    for scen_name, calib_names in SCENARIOS.items():
        dT_vals = []
        dR_vals = []
        reproj_vals = []
        pose_counts = []

        for cname in calib_names:
            res = results_by_name.get(cname)
            if res is None:
                continue

            dT_vals.append(res["mean_dT"])
            dR_vals.append(res["mean_dR"])
            reproj_vals.append(res["mean_reproj"])
            pose_counts.append(res["numPoses"])

        if not dT_vals:
            continue

        mean_dT = float(np.mean(dT_vals))
        mean_dR = float(np.mean(dR_vals))
        mean_reproj = float(np.mean(reproj_vals))
        mean_poses = float(np.mean(pose_counts))
        num_runs = len(dT_vals)

        print(
            f"{scen_name:<22} "
            f"{num_runs:6d} "
            f"{mean_poses:9.1f} "
            f"{mean_dT:12.5f} "
            f"{mean_dR:13.5f} "
            f"{mean_reproj:16.5f}"
        )

def main():
    results_by_name = {}

    for name, calib_path in CALIB_FILES.items():
        if not calib_path.exists():
            print(f"[WARN] File not found: {calib_path}")
            continue

        gt_path = GT_FILES.get(name)
        if gt_path is None:
            print(f"[WARN] No GT file mapped for {name}, skipping.")
            continue
        if not gt_path.exists():
            print(f"[WARN] GT file not found for {name}: {gt_path}")
            continue

        metrics = compute_metrics_for_file(calib_path, gt_path)
        results_by_name[name] = metrics

    if not results_by_name:
        print("[ERROR] No valid results computed. Check file paths.")
        return

    print("Detailed results per calibration:\n")
    print_per_file_table(results_by_name)

    print_scenario_table(results_by_name)


if __name__ == "__main__":
    main()