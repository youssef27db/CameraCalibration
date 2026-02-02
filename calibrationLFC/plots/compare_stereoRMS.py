import matplotlib
matplotlib.use("Agg")  # Headless mode for Docker/server environments

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths 
# This file's directory = calibrationLFC/plots
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# calibrationLFC directory
CALIB_LFC_DIR = os.path.dirname(THIS_DIR)

# Path to calibration result
CALIB_RESULT_PATH = os.path.join(
    CALIB_LFC_DIR, "results", "calibration_initial_imageset7_20251215_203849.json"
)

def load_stereo_rms(path):
    """Load stereo RMS values from calibration result."""
    with open(path, "r") as f:
        data = json.load(f)

    # Camera order as in meta block
    cam_ids = data["meta"]["cameraIds"]
    extrinsics = data["state"]["extrinsics"]

    cams = []
    rms_list = []

    for cam in cam_ids:
        if cam not in extrinsics:
            # Skip if no stereo calibration exists for this camera
            continue
        cams.append(cam)
        rms = float(extrinsics[cam].get("stereoRms", 0.0))
        rms_list.append(rms)

    return cams, np.array(rms_list)


def main():
    cams, rms_vals = load_stereo_rms(CALIB_RESULT_PATH)

    x = np.arange(len(cams))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        x, rms_vals,
        marker="o",
        linestyle="-",
        linewidth=2,
        color="#9467bd",  # slightly purple
        label="Stereo RMS"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cams, rotation=45)
    ax.set_ylabel("Stereo-Reprojection Error (Pixel)")
    ax.set_title("StereoRMS per Camera (OpenCV stereoCalibrate)")
    ax.grid(alpha=0.3)
    ax.legend()

    # Add some margin to y-axis
    ymin = max(0.0, rms_vals.min() - 1.0)
    ymax = rms_vals.max() + 1.0
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()

    # Save plot
    out_path = os.path.join(THIS_DIR, "compare_stereoRMS.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to:\n{out_path}")


if __name__ == "__main__":
    main()