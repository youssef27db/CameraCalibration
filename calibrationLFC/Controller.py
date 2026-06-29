from InitialCalibration import InitialCalibration
from ResultLogger import ResultLogger
from HealthMonitor import compute_initial_health_from_state


class Controller:
    """
    @brief Main controller for managing calibration workflow.
    
    @details Orchestrates the complete calibration process including initial calibration,
    health monitoring, and result logging. Manages the calibration state and maintains
    the history of health scores.
    """

    def __init__(self, bundleAdjust):
        """
        @brief Constructor for Controller.
        
        @param bundleAdjust Boolean flag indicating whether to apply bundle adjustment
        
        @details Initializes the controller with an InitialCalibration instance,
        empty score history, result logger, and bundle adjustment setting.
        """
        self.initialCalibration = InitialCalibration()
        self.scoreHistory = [] 
        self.currentState = None 
        self.resultLogger = ResultLogger()
        self.bundleAdjust = bundleAdjust 

    def runInitialCalibration(self, imageSet):
        """
        @brief Execute initial calibration for all cameras in the image set.
        
        @param imageSet ImageSet object containing calibration images and camera information
        @return CalibrationState object with computed intrinsic and extrinsic parameters
    
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
        @brief Computes a global health score for the calibration quality.
        
        @param calibrationState CalibrationState object to evaluate
        @return Health score as float between 0 and 100, or 0.0 if evaluation fails
        
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
