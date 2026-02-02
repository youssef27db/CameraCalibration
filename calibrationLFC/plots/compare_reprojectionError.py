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

def load_reprojection_errors(path):
    """Load reprojection errors from calibration result."""
    with open(path, "r") as f:
        data = json.load(f)

    intr = data["state"]["intrinsics"]

    cam_ids = []
    errors = []

    for cam, values in intr.items():
        cam_ids.append(cam)
        errors.append(float(values["reprojectionError"]))

    return cam_ids, np.array(errors)


def main():
    cams, errors = load_reprojection_errors(CALIB_RESULT_PATH)

    x = np.arange(len(cams))

    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(12, 5))

    plt.plot(
        x, errors,
        marker="o", linestyle="-", linewidth=2,
        color="#9d4edd",
        label="Reprojection Error"
    )

    plt.axhline(0, color="black", linewidth=1)

    plt.xticks(x, cams, rotation=45)
    plt.ylabel("Reprojection Error (pixels)")
    plt.title("Reprojection Error per Camera")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save plot
    out_path = os.path.join(THIS_DIR, "compare_reprojectionError.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to:\n{out_path}")


if __name__ == "__main__":
    main()
