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
    CALIB_LFC_DIR, "results", "calibration_initial_imageset6_20251211_040849.json"
)

def load_reprojection_errors(path):
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

    plt.figure(figsize=(12, 5))

    plt.plot(
        x, errors,
        marker="o", linestyle="-", linewidth=2,
        color="#9d4edd",  # schönes Lila
        label="Reprojection Error"
    )

    plt.axhline(0, color="black", linewidth=1)

    plt.xticks(x, cams, rotation=45)
    plt.ylabel("Reprojection Error (RMS)")
    plt.title("Reprojection Error pro Kamera")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
