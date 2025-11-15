import cv2
import numpy as np
from CalibrationState import CalibrationState

class InitialCalibration:

    def __init__(self, chessboardSize=(6,8), squareSize=0.02):
        self.chessboardSize = chessboardSize
        self.squareSize = squareSize
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # ---------------------------
    # Helper: object points, corner detection
    # ---------------------------
    def createObjectPoints(self):
        cols, rows = self.chessboardSize
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.squareSize
        return objp

    def detectCorners(self, imagePath):
        img = cv2.imread(imagePath)
        if img is None:
            return False, None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, self.chessboardSize)
        if not found:
            return False, None, None

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        return True, refined, gray.shape[::-1]

    # -------------------------------------------------------
    # Step 1: Intrinsic calibration per camera
    # -------------------------------------------------------
    def calibrateIntrinsics(self, imageSet):
        state = CalibrationState()
        objp = self.createObjectPoints()

        objpoints = {cam: [] for cam in imageSet.cameraIds}
        imgpoints = {cam: [] for cam in imageSet.cameraIds}
        imgsize = {}

        # collect points
        for pose in range(imageSet.numPoses):
            for cam in imageSet.cameraIds:
                path = imageSet.getImagePath(pose, cam)
                ok, corners, size = self.detectCorners(path)
                if ok:
                    objpoints[cam].append(objp)
                    imgpoints[cam].append(corners)
                    imgsize[cam] = size

        # calibrate each camera separately
        for cam in imageSet.cameraIds:
            if len(objpoints[cam]) < 5:
                print(f"[WARN] Not enough samples for camera {cam}")
                continue

            err, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints[cam], imgpoints[cam], imgsize[cam], None, None)

            state.setIntrinsics(cam, K, dist, err)

        return state

    # -------------------------------------------------------
    # Step 2: Extrinsic pairwise stereo calibration
    # -------------------------------------------------------
    def calibrateExtrinsics(self, imageSet, state, refCamId):
        objectPoints = self.createObjectPoints()
        K_ref = state.intrinsics[refCamId]["cameraMatrix"]
        dist_ref = state.intrinsics[refCamId]["distortionCoeffs"]

        for cam in imageSet.cameraIds:
            if cam == refCamId:
                state.setExtrinsics(cam, np.eye(3), np.zeros((3,1)), 0.0)
                continue

            if cam not in state.intrinsics:
                continue

            K_cam = state.intrinsics[cam]["cameraMatrix"]
            dist_cam = state.intrinsics[cam]["distortionCoeffs"]

            objList = []
            imgRef = []
            imgCam = []
            imageSize = None

            for pose in range(imageSet.numPoses):
                pRef = imageSet.getImagePath(pose, refCamId)
                pCam = imageSet.getImagePath(pose, cam)

                okR, cornersR, sizeR = self.detectCorners(pRef)
                okC, cornersC, sizeC = self.detectCorners(pCam)

                if okR and okC:
                    objList.append(objectPoints)
                    imgRef.append(cornersR)
                    imgCam.append(cornersC)
                    imageSize = sizeR

            if len(objList) < 3:
                continue

            flags = cv2.CALIB_FIX_INTRINSIC

            stereo_rms, KR0, d0, KC1, d1, R, T, E, F = cv2.stereoCalibrate(
                objList,
                imgRef,
                imgCam,
                K_ref,
                dist_ref,
                K_cam,
                dist_cam,
                imageSize,
                criteria=self.criteria,
                flags=flags
            )

            state.setExtrinsics(cam, R, T, stereo_rms)

        return state

    # -------------------------------------------------------
    # PUBLIC API: run(imageSet)
    # -------------------------------------------------------
    def run(self, imageSet):
        """
        Main entry point.
        Returns a filled CalibrationState.
        """
        print("Running intrinsic calibration...")
        state = self.calibrateIntrinsics(imageSet)

        print("\nRunning pairwise extrinsic calibration...")
        state = self.calibrateExtrinsics(imageSet, state, refCamId=imageSet.cameraIds[0])

        return state
