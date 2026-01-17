from InitialCalibration import InitialCalibration
from ResultLogger import ResultLogger
from HealthMonitor import compute_initial_health_from_state


class Controller:
    """
    Main controller for managing calibration workflow.
    Orchestrates initial calibration, health monitoring, and result logging.
    """

    def __init__(self, bundleAdjust):
        self.initialCalibration = InitialCalibration()
        self.scoreHistory = []
        self.currentState = None
        self.resultLogger = ResultLogger()
        self.bundleAdjust = bundleAdjust

    def runInitialCalibration(self, imageSet):
        """
        Execute initial calibration for all cameras in the image set.
        Logs results with metadata about the calibration run.
        """
        print("Controller: starting initial calibration...")
        calibrationState = self.initialCalibration.run(imageSet, self.bundleAdjust)

        # Prepare metadata for logging
        meta = {
            "runType": "initial",
            "numPoses": imageSet.numPoses,
            "imageDir": imageSet.baseDir,
            "cameraIds": imageSet.cameraIds,
            "bundleAdjust": self.bundleAdjust,
        }
        self.resultLogger.logInitialCalibration(calibrationState, meta=meta)

        return calibrationState

    def selfHealthCheck(self, calibrationState) -> float:
        """
        Computes a global health score (0..100) for the calibration quality.
        Based on reprojection errors and stereo RMS values.
        """
        # Extract state dictionary
        try:
            state_dict = calibrationState.__getState__()
        except Exception as e:
            self.resultLogger.logger.error(
                f"HealthCheck: could not extract state dict: {e}", exc_info=True
            )
            return 0.0

        # Compute health score from calibration metrics
        try:
            report = compute_initial_health_from_state(state_dict)
            score = float(report["health"])
            self.resultLogger.logger.info(
                f"HealthScore = {score:.2f} (method={report.get('method')})"
            )
            return score
        except Exception as e:
            self.resultLogger.logger.error(f"HealthCheck failed: {e}", exc_info=True)
            return 0.0
