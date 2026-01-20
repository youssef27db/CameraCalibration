import matplotlib
matplotlib.use("Agg")  # Headless mode for Docker/server environments

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Paths
# This file's directory:
# Data-Driven-Camera-Calibration/calibrationLFC/plots/LiFCal_Calibration
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root: Data-Driven-Camera-Calibration
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(THIS_DIR)
    )
)

# Paths to calibration results
INITIAL_CALIB_PATH = os.path.join(
    PROJECT_ROOT,
    "calibrationLFC",
    "results",
    "calibration_initial_imageset5_20251217_011656.json"
)

LIFCAL_CALIB_PATH = os.path.join(
    PROJECT_ROOT,
    "baseline",
    "calibration_lifcal_20260118_150049.json"
)


def load_initial_reprojection_error(path):
    """Load reprojectionError from initial calibration."""
    with open(path, "r") as f:
        data = json.load(f)

    intr = data["state"]["intrinsics"]

    cams = []
    errors = []

    for cam, values in intr.items():
        cams.append(cam)
        errors.append(float(values["reprojectionError"]))

    return cams, np.array(errors)


def load_lifcal_mean_std(path):
    """Load mean(x_std, y_std) from LiFCal calibration."""
    with open(path, "r") as f:
        data = json.load(f)

    params = data["state"]["parameters"]

    cams = []
    mean_std = []

    for cam, values in params.items():
        repro = values.get("reprojection", {})
        x_std = repro.get("x_std", np.nan)
        y_std = repro.get("y_std", np.nan)
        cams.append(cam)
        mean_std.append(np.mean([x_std, y_std]))

    return cams, np.array(mean_std)


def main():
    cams_init, init_errors = load_initial_reprojection_error(INITIAL_CALIB_PATH)
    cams_lif, lifcal_mean = load_lifcal_mean_std(LIFCAL_CALIB_PATH)

    # Safety check (order must match)
    assert cams_init == cams_lif, "Camera order mismatch between JSON files"

    x = np.arange(len(cams_init))

    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(13, 5))

    plt.plot(
        x, init_errors,
        marker="o", linestyle="-", linewidth=2,
        label="Initial Calibration (RMS)"
    )

    plt.plot(
        x, lifcal_mean,
        marker="s", linestyle="-", linewidth=2,
        label="LiFCal mean(x_std, y_std)"
    )

    plt.axhline(0, linewidth=1)

    plt.xticks(x, cams_init, rotation=45)
    plt.ylabel("Reprojection Error (Pixel)")
    plt.title("Reprojection Error Vergleich: Initial vs. LiFCal")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save plot
    out_path = os.path.join(
        THIS_DIR,
        "compare_initial_vs_lifcal_reprojection.png"
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to:\n{out_path}")


if __name__ == "__main__":
    main()
