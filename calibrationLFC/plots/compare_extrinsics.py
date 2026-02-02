import matplotlib
matplotlib.use("Agg")  # Headless mode for Docker/server environments

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Pfade 
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_LFC_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(CALIB_LFC_DIR)

CALIB_PATH = os.path.join(
    CALIB_LFC_DIR, "results",
    "calibration_initial_imageset7_20251215_203849.json"  
)

GT_PATH = os.path.join(
    PROJECT_ROOT, "TestEnvironment", "params",
    "groundtruth_extrinsics_Rig3_set7.json"
)

CAM_IDS = [
    "Center",
    "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3",
]


def rot_angle_deg(R):
    """Extract rotation angle (degrees) from 3x3 rotation matrix."""
    tr = np.trace(R)
    c = (tr - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def as_vec3(v):
    """Convert any translation vector (list/nested list) to 3-element vector."""
    arr = np.array(v, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"Expected 3 entries for translation, got {arr.size}")
    return arr


def main():
    # Load JSON files
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib = json.load(f)
    extr_est = calib["state"]["extrinsics"]

    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)

    rot_err_deg = []
    trans_err_abs = []
    trans_err_rel = []  # nur für Konsolen-Statistik

    for cam in CAM_IDS:
        if cam not in extr_est or cam not in gt:
            print(f"[WARN] Camera '{cam}' not in both files, skipping.")
            continue

        R_est = np.array(extr_est[cam]["rotationMatrix"], dtype=float)
        T_est = as_vec3(extr_est[cam]["translationVector"])

        R_gt = np.array(gt[cam]["rotationMatrix"], dtype=float)
        T_gt = as_vec3(gt[cam]["translationVector"]) 

        # --- Rotationsfehler ---
        R_err = R_est @ R_gt.T
        ang = rot_angle_deg(R_err)

        # --- Translationsfehler ---
        diff = T_est - T_gt
        d_abs = float(np.linalg.norm(diff))
        norm_gt = float(np.linalg.norm(T_gt))
        if norm_gt < 1e-8:
            d_rel = 0.0  # Center definieren wir als 0 %
        else:
            d_rel = 100.0 * d_abs / norm_gt

        rot_err_deg.append(ang)
        trans_err_abs.append(d_abs)
        trans_err_rel.append(d_rel)

    rot_err_deg = np.array(rot_err_deg)
    trans_err_abs = np.array(trans_err_abs)
    trans_err_rel = np.array(trans_err_rel)

    x = np.arange(len(CAM_IDS))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Plot 1: Rotation error
    ax1 = axes[0]
    ax1.bar(x, rot_err_deg, color="#7f3cff")
    ax1.set_xticks(x)
    ax1.set_xticklabels(CAM_IDS, rotation=45)
    ax1.set_ylabel("Rotation Error (°)")
    ax1.set_title("Rotation Error per Camera")

    mean_rot = rot_err_deg.mean()
    ax1.axhline(mean_rot, color="black", linestyle="--", linewidth=1)
    ax1.text(0.02, 0.95,
             f"Ø = {mean_rot:.2f}°",
             transform=ax1.transAxes,
             va="top", ha="left")

    # Plot 2: Translation error (meters)
    ax2 = axes[1]
    ax2.bar(x, trans_err_abs, color="#ff3ccf")
    ax2.set_xticks(x)
    ax2.set_xticklabels(CAM_IDS, rotation=45)
    ax2.set_ylabel("Translation Error (m)")
    ax2.set_title("Translation Error per Camera (absolute)")

    mean_abs = trans_err_abs.mean()
    ax2.axhline(mean_abs, color="black", linestyle="--", linewidth=1)
    ax2.text(0.02, 0.95,
             f"Ø = {mean_abs:.3f} m",
             transform=ax2.transAxes,
             va="top", ha="left")

    fig.suptitle("Extrinsics Quality: Ground-Truth vs. Calibration", fontsize=14)
    plt.tight_layout()

    # Save plot
    out_path = os.path.join(THIS_DIR, "compare_extrinsics.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to:\n{out_path}")
    print(f"Mean rotation error: {mean_rot:.2f}°")
    print(f"Mean translation error: {mean_abs:.3f} m")


if __name__ == "__main__":
    main()
