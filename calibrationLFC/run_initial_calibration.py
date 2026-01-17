"""
Test runner for initial calibration.

Usage:
  cd calibrationLFC
  python run_initial_calibration.py --base_dir /path/to/imageset --num_poses 28 --bundle-adjust
  python run_initial_calibration.py --base_dir /path/to/imageset --num_poses 28 --no-bundle-adjust

Performs validation checks (file existence, corner detection on pose 0)
then runs the complete initial calibration pipeline. Outputs summary statistics
for intrinsics, extrinsics, and health score.
"""

import argparse
import os
import sys

from ImageSet import ImageSet
from Controller import Controller


# Default configuration parameters
DEFAULT_BASE_DIR = "/data/calibrationLFC/sets/imageset5"
DEFAULT_NUM_POSES = 28
CAM_IDS = [
    "Center",
    "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run initial camera calibration")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR, help="Path to imageset directory")
    parser.add_argument("--num_poses", type=int, default=DEFAULT_NUM_POSES, help="Number of poses in the imageset")
    parser.add_argument(
        "--bundle-adjust",
        dest="bundle_adjust",
        action="store_true",
        help="Enable bundle adjustment (default)",
    )
    parser.add_argument(
        "--no-bundle-adjust",
        dest="bundle_adjust",
        action="store_false",
        help="Disable bundle adjustment",
    )
    parser.set_defaults(bundle_adjust=True)
    return parser.parse_args()


def main(args):
    """Run complete initial calibration pipeline with validation and summary output."""
    print("Creating ImageSet...")
    imageSet = ImageSet(args.base_dir, args.num_poses, CAM_IDS)

    print("Instantiating Controller and InitialCalibration...")
    controller = Controller(bundleAdjust=args.bundle_adjust)
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

    # Print calibration results per camera
    print("\nCalibration Results by Camera")
    print("=" * 50)

    for cam in CAM_IDS:
        print(f"\n--- {cam} ---")

        # Intrinsic parameters
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

        # Extrinsic parameters
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
            print("  Extrinsics: (no data)")

    # Print health score
    print("\nOverall Health Score")
    print("=" * 50)
    score = controller.selfHealthCheck(calibrationState)
    print(f"Global HealthScore = {score:.2f} / 100")

if __name__ == "__main__":
    # Set up Python path to find calibrationLFC modules
    repoRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repoRoot not in sys.path:
        sys.path.insert(0, repoRoot)
    main(parse_args())
