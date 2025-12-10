from InitialCalibration import InitialCalibration
from ResultLogger import ResultLogger

class Controller:

    def __init__(self, bundleAdjust):
        self.initialCalibration = InitialCalibration()
        self.scoreHistory = []
        self.currentState = None
        self.resultLogger = ResultLogger()
        self.bundleAdjust = bundleAdjust

    def runInitialCalibration(self, imageSet):
        print("Controller: starting initial calibration...")
        calibrationState = self.initialCalibration.run(imageSet, self.bundleAdjust)

        meta = {
            "runType": "initial",
            "numPoses": imageSet.numPoses,
            "imageDir": imageSet.baseDir,
            "cameraIds": imageSet.cameraIds,
            "bundleAdjust": self.bundleAdjust,
        }
        self.resultLogger.logInitialCalibration(calibrationState, meta=meta)

        return calibrationState
    
    def recalibration(self, imageSet):
        pass

    def selfHealthCheck(self, imageSet):
        pass
