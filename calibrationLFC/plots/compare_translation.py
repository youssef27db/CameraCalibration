import matplotlib
matplotlib.use("Agg")  # Headless mode for Docker/server environments

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

# Paths 
# This file's directory = calibrationLFC/plots
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# calibrationLFC directory
CALIB_LFC_DIR = os.path.dirname(THIS_DIR)

# Project root
PROJECT_ROOT = os.path.dirname(CALIB_LFC_DIR)

CALIB_PATH = os.path.join(
    CALIB_LFC_DIR,
    "results",
    "calibration_initial_imageset5_20251217_011656.json",
)

GT_PATH = os.path.join(
    PROJECT_ROOT,
    "TestEnvironment",
    "params",
    "groundtruth_extrinsics_Rig1_set5.json",
)

CAM_IDS = [
    "Center",
    "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3",
]

# ===================== Main =====================

def main():
    # Load JSON files
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib = json.load(f)

    extr_est = calib["state"]["extrinsics"]

    # Collect translation vectors
    T_gt = []
    T_est = []

    for cam in CAM_IDS:
        t_gt = np.array(gt[cam]["translationVector"], dtype=float).reshape(3)
        t_est = np.array(extr_est[cam]["translationVector"], dtype=float).reshape(3)

        T_gt.append(t_gt)
        T_est.append(t_est)

    T_gt = np.stack(T_gt, axis=0)
    T_est = np.stack(T_est, axis=0)

    # Calculate errors
    errors = np.linalg.norm(T_est - T_gt, axis=1)
    mean_error = np.mean(errors)

    # Extract axes
    X_gt, Y_gt, Z_gt = T_gt[:, 0], T_gt[:, 1], T_gt[:, 2]
    X_est, Y_est, Z_est = T_est[:, 0], T_est[:, 1], T_est[:, 2]

    def set_equal_3d(ax):
        """Set equal aspect ratio for 3D plot."""
        xs = np.concatenate([X_gt, X_est])
        ys = np.concatenate([Y_gt, Y_est])
        zs = np.concatenate([Z_gt, Z_est])

        max_range = max(xs.ptp(), ys.ptp(), zs.ptp())
        x_mid = xs.mean()
        y_mid = ys.mean()
        z_mid = zs.mean()

        ax.set_xlim(x_mid - max_range / 2, x_mid + max_range / 2)
        ax.set_ylim(y_mid - max_range / 2, y_mid + max_range / 2)
        ax.set_zlim(z_mid - max_range / 2, z_mid + max_range / 2)

    # Create plot
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(
        "Camera Positions in Space: Ground-Truth vs. Calibrated Extrinsics",
        fontsize=14,
    )

    # Plot 1: Groundtruth only
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(X_gt, Y_gt, Z_gt, c="black", s=40, label="Ground-Truth")

    for x, y, z, name in zip(X_gt, Y_gt, Z_gt, CAM_IDS):
        ax1.text(x, y, z, name, fontsize=8)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("Ground-Truth Camera Positions (3D)")
    set_equal_3d(ax1)
    ax1.legend()

    # Plot 2: Comparison
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax2.scatter(X_gt, Y_gt, Z_gt, c="black", s=40, label="Ground-Truth")
    ax2.scatter(X_est, Y_est, Z_est, c="#ff3ccf", s=40, label="Calibration")

    for xg, yg, zg, xe, ye, ze in zip(
        X_gt, Y_gt, Z_gt, X_est, Y_est, Z_est
    ):
        ax2.plot([xg, xe], [yg, ye], [zg, ze], color="#ff3ccf", alpha=0.7)

    for x, y, z, name in zip(X_est, Y_est, Z_est, CAM_IDS):
        ax2.text(x, y, z, name, fontsize=8)

    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    ax2.set_title("Ground-Truth vs. Calibrated Positions (3D)")
    set_equal_3d(ax2)
    ax2.legend()

    # Display mean error
    fig.text(
        0.70,
        0.01,
        f"Mean error: {mean_error:.3f} m",
        fontsize=12,
        ha="center",
    )

    plt.tight_layout()

    # Save plot
    out_path = os.path.join(THIS_DIR, "compare_translation.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to:\n{out_path}")
    print(f"Mean translation error: {mean_error:.3f} m")


if __name__ == "__main__":
    main()
