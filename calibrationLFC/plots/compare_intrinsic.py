import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Dieser Dateiordner = calibrationLFC/plots
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# calibrationLFC-Ordner
CALIB_LFC_DIR = os.path.dirname(THIS_DIR)

# Projektroot = eine Ebene höher (CameraCalibration)
PROJECT_ROOT = os.path.dirname(CALIB_LFC_DIR)

# Pfad zur Groundtruth-Datei
GROUNDTRUTH_PATH = os.path.join(
    PROJECT_ROOT, "TestEnvironment", "params", "groundTruth_ideal.json"
)

# Pfad zum Kalibrierungsergebnis
CALIB_RESULT_PATH = os.path.join(
    CALIB_LFC_DIR, "results", "calibration_initial_imageset1_20251210_171918.json"
)


def load_groundtruth_intrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)
    intr = data["intrinsics"]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    return fx, fy, cx, cy


def load_calibrated_intrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)

    intr_dict = data["state"]["intrinsics"]

    cams = []
    fx_list, fy_list, cx_list, cy_list = [], [], [], []

    for cam_id, cam_data in intr_dict.items():
        K = cam_data["cameraMatrix"]  # 3x3
        cams.append(cam_id)
        fx_list.append(float(K[0][0]))
        fy_list.append(float(K[1][1]))
        cx_list.append(float(K[0][2]))
        cy_list.append(float(K[1][2]))

    return cams, np.array(fx_list), np.array(fy_list), np.array(cx_list), np.array(cy_list)


def main():
    fx_gt, fy_gt, cx_gt, cy_gt = load_groundtruth_intrinsics(GROUNDTRUTH_PATH)
    cams, fx_est, fy_est, cx_est, cy_est = load_calibrated_intrinsics(CALIB_RESULT_PATH)

    # Differenzen (est - gt)
    d_fx = fx_est - fx_gt
    d_fy = fy_est - fy_gt
    d_cx = cx_est - cx_gt
    d_cy = cy_est - cy_gt

    x = np.arange(len(cams))

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    # ------------------ Plot 1: Brennweiten (fx, fy) ------------------
    ax1 = axes[0]
    ax1.plot(x, d_fx, marker="o", linestyle="-", linewidth=2,
             color="#1f77b4", label="Δfx")
    ax1.plot(x, d_fy, marker="s", linestyle="--", linewidth=2,
             color="#ff7f0e", label="Δfy")

    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cams, rotation=45)
    ax1.set_ylabel("Differenz (Pixel)")
    ax1.set_title("Abweichung Brennweiten (fx, fy)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ------------------ Plot 2: Hauptpunkt (cx, cy) -------------------
    ax2 = axes[1]
    ax2.plot(x, d_cx, marker="^", linestyle="-", linewidth=2,
             color="#2ca02c", label="Δcx")
    ax2.plot(x, d_cy, marker="D", linestyle="--", linewidth=2,
             color="#d62728", label="Δcy")

    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cams, rotation=45)
    ax2.set_ylabel("Differenz (Pixel)")
    ax2.set_title("Abweichung Hauptpunkt (cx, cy)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Abweichung der geschätzten Intrinsics gegenüber Ground-Truth", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
