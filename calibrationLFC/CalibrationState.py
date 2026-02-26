class CalibrationState:
    """
    @brief Stores the complete calibration state for a multi-camera system.
    
    @details This class manages intrinsic and extrinsic parameters for all cameras
    in the system. It provides methods to set and retrieve calibration data including
    camera matrices, distortion coefficients, rotation matrices, and translation vectors.
    """
    
    def __init__(self):
        """
        @brief Constructor for CalibrationState.
        
        @details Initializes empty dictionaries for intrinsic and extrinsic parameters
        and sets the timestamp to None.
        """
        self.intrinsics = {} ## Dictionary: cameraId -> {cameraMatrix, distortionCoeffs, reprojectionError}
        self.extrinsics = {} ## Dictionary: cameraId -> {rotationMatrix, translationVector, stereoRms}
        self.timeStamp = None ## Timestamp of the calibration

    def setIntrinsics(self, camId, cameraMatrix, distortionCoeffs, repError):
        """
        @brief Store intrinsic calibration parameters for a camera.
        
        @param camId Camera identifier
        @param cameraMatrix 3x3 camera matrix containing focal lengths and principal point
        @param distortionCoeffs Distortion coefficients vector
        @param repError Reprojection error of the calibration
        """
        self.intrinsics[camId] = {
            "cameraMatrix": cameraMatrix,
            "distortionCoeffs": distortionCoeffs,
            "reprojectionError": repError
        }

    def setExtrinsics(self, camId, rotationMatrix, translationVector, stereoRMS):
        """
        @brief Store extrinsic calibration parameters for a camera.
        
        @param camId Camera identifier
        @param rotationMatrix 3x3 rotation matrix representing camera orientation
        @param translationVector 3x1 translation vector representing camera position
        @param stereoRMS Root mean square error of the stereo calibration
        """
        self.extrinsics[camId] = {
            "rotationMatrix": rotationMatrix,
            "translationVector": translationVector,
            "stereoRms": stereoRMS
        }

    def setTimeStamp(self, timestamp):
        """
        @brief Set the calibration timestamp.
        
        @param timestamp Timestamp indicating when the calibration was performed
        """
        self.timeStamp = timestamp

    def __getState__(self):
        """
        @brief Return the complete calibration state as dictionary.
        
        @return Dictionary containing intrinsics, extrinsics, and timestamp
        """
        return {
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
            "timeStamp": self.timeStamp
        }
