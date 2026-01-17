class CalibrationState:
    """
    Stores the complete calibration state for a multi-camera system.
    Manages intrinsic and extrinsic parameters for all cameras.
    """
    
    def __init__(self):
        self.intrinsics = {} # Dictionary: cameraId -> {cameraMatrix, distortionCoeffs, reprojectionError}
        self.extrinsics = {} # Dictionary: cameraId -> {rotationMatrix, translationVector, stereoRms}
        self.timeStamp = None

    def setIntrinsics(self, camId, cameraMatrix, distortionCoeffs, repError):
        """Store intrinsic calibration parameters for a camera."""
        self.intrinsics[camId] = {
            "cameraMatrix": cameraMatrix,
            "distortionCoeffs": distortionCoeffs,
            "reprojectionError": repError
        }

    def setExtrinsics(self, camId, rotationMatrix, translationVector, stereoRMS):
        """Store extrinsic calibration parameters for a camera."""
        self.extrinsics[camId] = {
            "rotationMatrix": rotationMatrix,
            "translationVector": translationVector,
            "stereoRms": stereoRMS
        }

    def setTimeStamp(self, timestamp):
        """Set the calibration timestamp."""
        self.timeStamp = timestamp

    def __getState__(self):
        """Return the complete calibration state as dictionary."""
        return {
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "timeStamp": self.timeStamp
        }
