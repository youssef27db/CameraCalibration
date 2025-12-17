"""
Small test runner for initial calibration.

How to run (from repository root):
  cd calibrationLFC
  python run_initial_calibration.py

This script performs quick checks (file existence, corner detection on pose 0)
and then calls the controller to run the initial calibration. It prints a short
summary at the end.
"""

import os
import sys

from ImageSet import ImageSet
from Controller import Controller


# -------------------------
# Configuration
# -------------------------
BASE_DIR = "imageset5"  # folder with images
NUM_POSES = 28
CAM_IDS = [
    "Center",
    "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3",
]


def main():
    print("Creating ImageSet...")
    imageSet = ImageSet(BASE_DIR, NUM_POSES, CAM_IDS)

    print("Instantiating Controller and InitialCalibration...")
    controller = Controller(bundleAdjust = False)
    logger = controller.resultLogger

    print("\nQuick checks (pose 0): file exists / corners detected")
    for cam in CAM_IDS:
        path = imageSet.getImagePath(0, cam)
        exists = os.path.exists(path)
        cornersFound = None

        try:
            cornersFound, _, _ = controller.initialCalibration.detectCorners(path)
        except Exception as e:
            cornersFound = f"error: {e.__class__.__name__}"
        print(f" {cam}: {path} - exists={exists} - corners={cornersFound}")

    print("\nRunning initial calibration (this may take time depending on data)...")
    try:
        calibrationState = controller.runInitialCalibration(imageSet)
    except Exception as e:
        logger.logger.error(f"Calibration failed with exception: {e}", exc_info=True)
        print("Calibration failed, see calibration.log for details.")
        raise

    print("\nCalibration finished. Attempting to print summary...")
    try:
        stateDict = calibrationState.__getState__()
    except Exception:
        stateDict = None

    if stateDict is None:
        print("Could not extract a state dictionary.")
        return

    intr = stateDict.get("intrinsics", {})
    extr = stateDict.get("extrinsics", {})

    print("\n==============================")
    print("   Calibration Summary")
    print("==============================")

    for cam in CAM_IDS:
        print(f"\n--- {cam} ---")

        # ===== Intrinsics =====
        if cam in intr:
            data = intr[cam]
            K = data["cameraMatrix"]
            dist = data["distortionCoeffs"].ravel()
            err = data["reprojectionError"]

            print("Intrinsics:")
            print(" K =")
            print(f"  {K[0][0]:.3f} {K[0][1]:.3f} {K[0][2]:.3f}")
            print(f"  {K[1][0]:.3f} {K[1][1]:.3f} {K[1][2]:.3f}")
            print(f"  {K[2][0]:.3f} {K[2][1]:.3f} {K[2][2]:.3f}")
            print(" distCoeffs =", dist)
            print(f" reprojectionError = {err:.6f}")
        else:
            print("Intrinsics: (no data)")

        # ===== Extrinsics =====
        if cam in extr:
            data = extr[cam]
            R = data["rotationMatrix"]
            T = data["translationVector"].ravel()
            rms = data["stereoRms"]

            print("Extrinsics:")
            print(" R =")
            print(f"  {R[0][0]:.5f} {R[0][1]:.5f} {R[0][2]:.5f}")
            print(f"  {R[1][0]:.5f} {R[1][1]:.5f} {R[1][2]:.5f}")
            print(f"  {R[2][0]:.5f} {R[2][1]:.5f} {R[2][2]:.5f}")
            print(" T =", T)
            print(f" stereoRMS = {rms:.6f}")
        else:
            print("Extrinsics: (no data)")



if __name__ == "__main__":
    # If the script is executed from repository root, adjust sys.path so imports work
    repoRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repoRoot not in sys.path:
        sys.path.insert(0, repoRoot)
    main()
