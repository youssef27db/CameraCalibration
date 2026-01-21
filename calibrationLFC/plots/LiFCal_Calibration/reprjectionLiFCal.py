import matplotlib
matplotlib.use("Agg")  # Headless mode for Docker/server environments

import json
import os
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Data-Driven-Camera-Calibration directory
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(THIS_DIR)
    )
)

# Path to calibration result
CALIB_RESULT_PATH = os.path.join(
    PROJECT_ROOT,
    "baseline",
    "calibration_lifcal_20260118_150049.json"
)

def load_reprojection_std(path):
    """Load x_std and y_std reprojection values per camera."""
    with open(path, "r") as f:
        data = json.load(f)

    params = data["state"]["parameters"]

    cam_ids = []
    x_std = []
    y_std = []

    for cam, values in params.items():
        repro = values.get("reprojection", {})
        cam_ids.append(cam)
        x_std.append(float(repro.get("x_std", np.nan)))
        y_std.append(float(repro.get("y_std", np.nan)))

    return cam_ids, np.array(x_std), np.array(y_std)


def main():
    cams, x_std, y_std = load_reprojection_std(CALIB_RESULT_PATH)

    x = np.arange(len(cams))

    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(12, 5))

    plt.plot(
        x, x_std,
        marker="o", linestyle="-", linewidth=2,
        label="x_std"
    )

    plt.plot(
        x, y_std,
        marker="s", linestyle="-", linewidth=2,
        label="y_std"
    )

    plt.axhline(0, linewidth=1)

    plt.xticks(x, cams, rotation=45)
    plt.ylabel("Reprojection Std (Pixel)")
    plt.title("Reprojection Standard Deviation per Camera")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save plot
    out_path = os.path.join(THIS_DIR, "reprojection_std_per_camera.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to:\n{out_path}")


if __name__ == "__main__":
    main()
