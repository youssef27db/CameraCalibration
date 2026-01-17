from typing import List

class ImageSet:
    """
    Represents a set of calibration images organized by pose and camera.
    Provides utilities for constructing image file paths.
    """
    
    def __init__(self, baseDir: str, numPoses: int, cameraIds: List[str]):
        self.baseDir = baseDir
        self.numPoses = numPoses
        self.cameraIds = cameraIds

    def getImagePath(self, poseIndex: int, cameraId: str) -> str:
        poseStr = f"pose_{poseIndex:03d}"
        return f"{self.baseDir}/{poseStr}_{cameraId}.png"
    
        """ 
            Beispiel:
            baseDir = "/data/images"
            poseIndex = 5  → poseStr = "pose_005"
            cameraId = "center"
            
            Ergebnis: "/data/images/pose_005_center.png" 
        """
