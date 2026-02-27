import matplotlib
matplotlib.use("Agg")

import json
import os
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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
    with open(path, "r") as f:
        data = json.load(f)

    params = data["state"]["parameters"]

    cam_order = data.get("meta", {}).get("cameraIds", sorted(params.keys()))

    x_std = []
    y_std = []

    for cam in cam_order:
        repro = params.get(cam, {}).get("reprojection", {})
        x_std.append(float(repro.get("x_std", np.nan)))
        y_std.append(float(repro.get("y_std", np.nan)))

    return cam_order, np.array(x_std), np.array(y_std)


def align_to_reference(ref_cams, cams, x_std, y_std):
    idx = {c: i for i, c in enumerate(cams)}
    x_aligned = np.array([x_std[idx[c]] if c in idx else np.nan for c in ref_cams])
    y_aligned = np.array([y_std[idx[c]] if c in idx else np.nan for c in ref_cams])
    return x_aligned, y_aligned


def combined_error(x_std, y_std):
    return np.sqrt(x_std**2 + y_std**2)


def main():
    ref_cams, _, _ = load_reprojection_std(RUNS[0]["path"])

    run_data = []
    combined_all = []

    for r in RUNS:
        cams, x_std, y_std = load_reprojection_std(r["path"])
        x_std, y_std = align_to_reference(ref_cams, cams, x_std, y_std)

        run_data.append((r["label"], x_std, y_std))
        combined_all.append(combined_error(x_std, y_std))

    combined_all = np.concatenate(combined_all)

    global_mean = float(np.nanmean(combined_all))
    global_std = float(np.nanstd(combined_all))

    x = np.arange(len(ref_cams))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    markers = ["o", "s", "^"]

    # --- X_STD ---
    ax = axes[0]
    for i, (label, x_std, _) in enumerate(run_data):
        ax.plot(x, x_std, marker=markers[i], linewidth=2, label=label)

    ax.set_ylabel("x_std (Pixel)")
    ax.set_title("Reprojection Std Comparison per Camera")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- Y_STD ---
    ax = axes[1]
    for i, (label, _, y_std) in enumerate(run_data):
        ax.plot(x, y_std, marker=markers[i], linewidth=2, label=label)

    ax.set_ylabel("y_std (Pixel)")
    ax.grid(alpha=0.3)
    ax.legend()

    # -------- CENTERED GLOBAL STAT TEXT --------
    summary_text = (
        "Combined mean Reprojection Error (3 runs)\n"
        rf"$Ø$ = {global_mean:.2f} ± {global_std:.2f} px"
        )

    axes[1].text(
        0.5, 0.95,  # centered
        summary_text,
        transform=axes[1].transAxes,
        fontsize=14,
        horizontalalignment="center",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9,
            edgecolor="black"
        )
    )

    plt.xticks(x, ref_cams, rotation=45)
    plt.tight_layout()

    out_path = os.path.join(THIS_DIR, "reprojection_std_compare_3runs.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Global Combined Reprojection Error:")
    print(f"{global_mean:.4f} ± {global_std:.4f} px")


if __name__ == "__main__":
    main()