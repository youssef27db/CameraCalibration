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

BASELINE_DIR = os.path.join(PROJECT_ROOT, "baseline")

RUNS = [
    {
        "label": "run1",
        "path": os.path.join(BASELINE_DIR, "calibration_lifcal_20260115_204926.json"),
    },
    {
        "label": "run2",
        "path": os.path.join(BASELINE_DIR, "calibration_lifcal_20260118_150049.json"),
    },
    {
        "label": "run3",
        "path": os.path.join(BASELINE_DIR, "calibration_lifcal_20260121_150131.json"),
    },
]


def load_reprojection_std(path):
    """Load x_std and y_std reprojection values per camera (ordered by meta.cameraIds if present)."""
    with open(path, "r") as f:
        data = json.load(f)

    params = data["state"]["parameters"]

    # Prefer stable camera order from meta if available
    cam_order = None
    if isinstance(data.get("meta", {}), dict):
        cam_order = data["meta"].get("cameraIds", None)

    if cam_order is None:
        cam_order = sorted(list(params.keys()))

    x_std = []
    y_std = []

    for cam in cam_order:
        values = params.get(cam, {})
        repro = values.get("reprojection", {}) if isinstance(values, dict) else {}
        x_std.append(float(repro.get("x_std", np.nan)))
        y_std.append(float(repro.get("y_std", np.nan)))

    return cam_order, np.array(x_std), np.array(y_std)


def align_to_reference(ref_cams, cams, x_std, y_std):
    """Align arrays (cams, x_std, y_std) to ref_cams ordering."""
    idx = {c: i for i, c in enumerate(cams)}
    x_aligned = np.array([x_std[idx[c]] if c in idx else np.nan for c in ref_cams], dtype=float)
    y_aligned = np.array([y_std[idx[c]] if c in idx else np.nan for c in ref_cams], dtype=float)
    return x_aligned, y_aligned


def main():
    # Load first run as reference order
    ref_cams, x0, y0 = load_reprojection_std(RUNS[0]["path"])

    run_data = []
    for r in RUNS:
        cams, x_std, y_std = load_reprojection_std(r["path"])
        x_std, y_std = align_to_reference(ref_cams, cams, x_std, y_std)
        run_data.append((r["label"], x_std, y_std))

    x = np.arange(len(ref_cams))

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    markers = ["o", "s", "^", "D", "v", "P", "X"]

    # --- X_STD plot ---
    ax = axes[0]
    for i, (label, x_std, _y_std) in enumerate(run_data):
        ax.plot(
            x, x_std,
            marker=markers[i % len(markers)],
            linestyle="-",
            linewidth=2,
            label=label
        )
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("x_std (Pixel)")
    ax.set_title("Reprojection Std Comparison (x_std) per Camera")
    ax.grid(alpha=0.3)
    ax.legend()

    # --- Y_STD plot ---
    ax = axes[1]
    for i, (label, _x_std, y_std) in enumerate(run_data):
        ax.plot(
            x, y_std,
            marker=markers[i % len(markers)],
            linestyle="-",
            linewidth=2,
            label=label
        )
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("y_std (Pixel)")
    ax.set_title("Reprojection Std Comparison (y_std) per Camera")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.xticks(x, ref_cams, rotation=45)
    plt.tight_layout()

    out_path = os.path.join(THIS_DIR, "reprojection_std_compare_3runs.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to:\n{out_path}\n")

    # Quick summary
    for label, x_std, y_std in run_data:
        print(
            f"{label}: "
            f"mean(x_std)={np.nanmean(x_std):.4f}, "
            f"mean(y_std)={np.nanmean(y_std):.4f}, "
            f"max(x_std)={np.nanmax(x_std):.4f}, "
            f"max(y_std)={np.nanmax(y_std):.4f}"
        )


if __name__ == "__main__":
    main()
