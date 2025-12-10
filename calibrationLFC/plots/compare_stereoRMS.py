import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Dieser Dateiordner = calibrationLFC/plots
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# calibrationLFC-Ordner
CALIB_LFC_DIR = os.path.dirname(THIS_DIR)

# Pfad zum Kalibrierungsergebnis
CALIB_RESULT_PATH = os.path.join(
    CALIB_LFC_DIR, "results", "calibration_initial_imageset1_20251210_171918.json"
)

def load_stereo_rms(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Kamerareihenfolge wie im Meta-Block
    cam_ids = data["meta"]["cameraIds"]
    extrinsics = data["state"]["extrinsics"]

    cams = []
    rms_list = []

    for cam in cam_ids:
        if cam not in extrinsics:
            # Falls für eine Kamera keine Stereo-Kalibrierung existiert, überspringen
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
        color="#9467bd",  # leicht violett
        label="Stereo RMS"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cams, rotation=45)
    ax.set_ylabel("Stereo-Reprojection Error (Pixel)")
    ax.set_title("StereoRMS pro Kamera (OpenCV stereoCalibrate)")
    ax.grid(alpha=0.3)
    ax.legend()

    # Optional: y-Achse etwas Luft nach oben/unten
    ymin = max(0.0, rms_vals.min() - 1.0)
    ymax = rms_vals.max() + 1.0
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()