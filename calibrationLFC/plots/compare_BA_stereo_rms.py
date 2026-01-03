import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Resolve results paths relative to repository root (two levels up from this script)
ROOT = Path(__file__).resolve().parents[1]
PATH_NO_BA = ROOT/"results"/"calibration_initial_imageset5_20251217_015917.json"  # ohne bundleAdjust
PATH_BA    = ROOT/"results"/"calibration_initial_imageset5_20251217_011656.json"  # mit bundleAdjust


def load_stereo_rms(path):
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] Results file not found: {path}")
        sys.exit(2)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    extr = data["state"]["extrinsics"]

    # gleiche Reihenfolge der Kameras
    cams = [c for c in [
        "Center", "Up1", "Up2", "Up3",
        "Down1", "Down2", "Down3",
        "Left1", "Left2", "Left3",
        "Right1", "Right2", "Right3",
    ] if c in extr]

    rms = np.array([extr[c]["stereoRms"] for c in cams], dtype=float)
    return cams, rms


def main():
    cams1, rms_no_ba = load_stereo_rms(PATH_NO_BA)
    cams2, rms_ba    = load_stereo_rms(PATH_BA)
    assert cams1 == cams2
    cams = cams1

    x = np.arange(len(cams))
    w = 0.35

    mean_no_ba = float(rms_no_ba.mean())
    mean_ba    = float(rms_ba.mean())
    diff       = mean_ba - mean_no_ba

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Vergleich stereoRMS (mit vs. ohne Bundle Adjustment)")

    # ----------------- Plot 1: stereoRMS pro Kamera -----------------
    ax1.bar(x - w/2, rms_no_ba, width=w, label=f"ohne BA (Ø {mean_no_ba:.2f}px)")
    ax1.bar(x + w/2, rms_ba,    width=w, label=f"mit BA (Ø {mean_ba:.2f}px)")

    # Mittelwerte als horizontale Linien einzeichnen
    ax1.axhline(mean_no_ba, color="C0", linestyle="--", alpha=0.7)
    ax1.axhline(mean_ba,    color="C1", linestyle="--", alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cams, rotation=45)
    ax1.set_ylabel("stereoRMS (px)")
    ax1.set_title("stereoRMS je Kamera")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ----------------- Plot 2: nur Mittelwerte -----------------
    ax2.bar([0], [mean_no_ba], width=0.5, label="ohne BA")
    ax2.bar([1], [mean_ba],    width=0.5, label="mit BA")

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["ohne BA", "mit BA"])
    ax2.set_ylabel("Durchschnitt stereoRMS (px)")
    ax2.set_title("Durchschnittlicher stereoRMS")

    # Werte als Text über die Balken schreiben
    ax2.text(0, mean_no_ba + 0.5, f"{mean_no_ba:.2f}px", ha="center", va="bottom")
    ax2.text(1, mean_ba + 0.5,    f"{mean_ba:.2f}px",    ha="center", va="bottom")

    # Optionale Info zum Unterschied
    ax2.text(0.5, max(mean_no_ba, mean_ba) + 2.0,
             f"Δ Ø = {diff:+.2f}px",
             ha="center", va="bottom")

    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Durchschnitt stereoRMS ohne BA:", mean_no_ba)
    print("Durchschnitt stereoRMS mit BA:", mean_ba)
    print("Differenz (mit - ohne):", diff)


if __name__ == "__main__":
    main()
