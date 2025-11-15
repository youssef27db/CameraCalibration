class CalibrationState:
    def __init__(self):
        self.intrinsics = {}   # cam -> {cameraMatrix, distortionCoeffs, reprojectionError}
        self.extrinsics = {}   # cam -> {rotationMatrix, translationVector, stereoRms}
        self.timeStamp = None

    def setIntrinsics(self, camId, cameraMatrix, distortionCoeffs, repError):
        self.intrinsics[camId] = {
            "cameraMatrix": cameraMatrix,
            "distortionCoeffs": distortionCoeffs,
            "reprojectionError": repError
        }

    def setExtrinsics(self, camId, rotationMatrix, translationVector, stereoRMS):
        self.extrinsics[camId] = {
            "rotationMatrix": rotationMatrix,
            "translationVector": translationVector,
            "stereoRms": stereoRMS
        }

    def setTimeStamp(self, timestamp):
        self.timeStamp = timestamp

    def __getState__(self):
        return {
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "timeStamp": self.timeStamp
        }
