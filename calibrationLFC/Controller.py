from InitialCalibration import InitialCalibration

class Controller:

    def __init__(self):
        self.initialCalibration = InitialCalibration()
        self.scoreHistory = []
        self.currentState = None

    def runInitialCalibration(self, imageSet):
        print("Controller: starting initial calibration...")
        calibrationState = self.initialCalibration.run(imageSet)
        return calibrationState
    
    def recalibration(self, imageSet):
        pass

    def selfHealthCheck(self, imageSet):
        pass
