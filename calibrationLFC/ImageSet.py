from typing import List

class ImageSet:
    """
    @brief Represents a set of calibration images organized by pose and camera.
    """
    
    def __init__(self, baseDir: str, numPoses: int, cameraIds: List[str]):
        """
        @brief Constructor for ImageSet.
        
        @param baseDir Base directory containing calibration images
        @param numPoses Total number of poses in the calibration set
        @param cameraIds List of camera identifiers
        """
        self.baseDir = baseDir
        self.numPoses = numPoses
        self.cameraIds = cameraIds

    def getImagePath(self, poseIndex: int, cameraId: str) -> str:
        """
        @brief Construct the file path for a specific pose and camera.
        
        @param poseIndex Zero-based pose index
        @param cameraId Camera identifier string
        @return String path in format: baseDir/pose_XXX_cameraId.png
        """
        poseStr = f"pose_{poseIndex:03d}"
        return f"{self.baseDir}/{poseStr}_{cameraId}.png"
    
        """ 
            Beispiel:
            baseDir = "/data/images"
            poseIndex = 5  → poseStr = "pose_005"
            cameraId = "center"
            
            Ergebnis: "/data/images/pose_005_center.png" 
        """
