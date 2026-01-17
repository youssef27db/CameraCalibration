import cv2
import numpy as np
from CalibrationState import CalibrationState
from scipy.optimize import least_squares


class InitialCalibration:
    """Multi-camera calibration using chessboard patterns and bundle adjustment."""

    def __init__(self, chessboardSize=(8,8), squareSize=2/9):
        self.chessboardSize = chessboardSize
        self.squareSize = squareSize
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def createObjectPoints(self):
        """Generate 3D coordinates of chessboard corners."""
        cols, rows = self.chessboardSize
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= self.squareSize
        return objp

    def detectCorners(self, imagePath):
        """Detect and refine chessboard corner positions in image."""
        img = cv2.imread(imagePath)
        if img is None:
            return False, None, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, self.chessboardSize)
        if not found:
            return False, None, None

        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        return True, refined, gray.shape[::-1]

    def calibrateIntrinsics(self, imageSet):
        """Calibrate intrinsic parameters independently for each camera."""
        state = CalibrationState()
        objp = self.createObjectPoints()

        objpoints = {cam: [] for cam in imageSet.cameraIds}
        imgpoints = {cam: [] for cam in imageSet.cameraIds}
        imgsize = {}

        # Collect corner points from all poses for each camera
        for pose in range(imageSet.numPoses):
            for cam in imageSet.cameraIds:
                path = imageSet.getImagePath(pose, cam)
                ok, corners, size = self.detectCorners(path)
                if ok:
                    objpoints[cam].append(objp)
                    imgpoints[cam].append(corners)
                    imgsize[cam] = size

        # Calibrate each camera separately
        for cam in imageSet.cameraIds:
            if len(objpoints[cam]) < 5:
                print(f"[WARN] Not enough samples for camera {cam}")
                continue

            err, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints[cam], imgpoints[cam], imgsize[cam], None, None)

            state.setIntrinsics(cam, K, dist, err)

        return state

    def calibrateExtrinsics(self, imageSet, state, refCamId):
        """Compute pairwise stereo calibration between reference and each other camera."""
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
    
    def bundleAdjustExtrinsics(self, imageSet, state, refCamId):
        """
        Global bundle adjustment optimizing extrinsics and board poses.
        
        Intrinsics remain fixed (from state.intrinsics).
        Optimizes:
          * Extrinsics of non-reference cameras (R_cam, t_cam)
          * Board pose per image (R_board, t_board) in reference camera frame
        """
        objp = self.createObjectPoints()
        cameraIds = list(imageSet.cameraIds)
        if refCamId not in cameraIds:
            raise ValueError("refCamId not in imageSet.cameraIds")

        ref_idx = cameraIds.index(refCamId)

        # Initialize board poses via PnP-RANSAC in reference camera
        K_ref = state.intrinsics[refCamId]["cameraMatrix"]
        dist_ref = state.intrinsics[refCamId]["distortionCoeffs"]

        pose_rvecs = {}
        pose_tvecs = {}
        valid_poses = []

        for pose in range(imageSet.numPoses):
            img_path = imageSet.getImagePath(pose, refCamId)
            ok, corners, size = self.detectCorners(img_path)
            if not ok:
                continue

            # Use PnP-RANSAC for robust board pose estimation
            ok_pnp, rvec, tvec, inliers = cv2.solvePnPRansac(
                objp,
                corners,
                K_ref,
                dist_ref,
                reprojectionError=2.0,   
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not ok_pnp or inliers is None or len(inliers) < 10:
                # Skip pose if too few reliable points
                continue

            pose_rvecs[pose] = rvec.astype(np.float64).reshape(3)
            pose_tvecs[pose] = tvec.astype(np.float64).reshape(3)
            valid_poses.append(pose)

        if len(valid_poses) < 3:
            print("[BA] Not enough valid poses for bundle adjustment.")
            return state

        # Initialize extrinsics from state
        # Only optimize non-reference cameras
        optim_cams = []
        cam_param_index = {}

        for ci, cam in enumerate(cameraIds):
            if cam == refCamId:
                continue
            if cam not in state.extrinsics:
                continue

            optim_cams.append(ci)
            cam_param_index[ci] = len(cam_param_index)

        num_optim_cams = len(optim_cams)
        if num_optim_cams == 0:
            print("[BA] No extrinsic data to optimize, skipping BA.")
            return state

        # Collect measurements (all cameras, all valid poses)
        # Each measurement: (cam_idx, pose, 2D points)
        measurements = []

        for ci, cam in enumerate(cameraIds):
            for pose in valid_poses:
                img_path = imageSet.getImagePath(pose, cam)
                ok, corners, size = self.detectCorners(img_path)
                if not ok:
                    continue
                pts2d = corners.reshape(-1, 2).astype(np.float64)
                measurements.append((ci, pose, pts2d))

        if len(measurements) == 0:
            print("[BA] No overlapping detections found, skipping BA.")
            return state

        # Initialize parameter vector
        # Layout: [cams (without ref): (r_cam(3), t_cam(3))... , poses: (r_p(3), t_p(3))...]

        param_cam = []
        for ci in optim_cams:
            cam = cameraIds[ci]
            extr = state.extrinsics[cam]
            R = extr["rotationMatrix"]
            T = extr["translationVector"].reshape(3)
            rvec_cam, _ = cv2.Rodrigues(R)
            param_cam.append(rvec_cam.reshape(3))
            param_cam.append(T)
        param_cam = np.concatenate(param_cam).astype(np.float64)

        param_pose = []
        for pose in valid_poses:
            param_pose.append(pose_rvecs[pose])
            param_pose.append(pose_tvecs[pose])
        param_pose = np.concatenate(param_pose).astype(np.float64)

        x0 = np.concatenate([param_cam, param_pose])

        # Helper functions for parameter decoding
        def decode_params(params):
            # Cameras
            cam_rvecs = {}
            cam_tvecs = {}
            idx = 0
            for ci in optim_cams:
                r = params[idx:idx+3]; idx += 3
                t = params[idx:idx+3]; idx += 3
                cam_rvecs[ci] = r
                cam_tvecs[ci] = t

            # Poses
            pose_r = {}
            pose_t = {}
            for pose in valid_poses:
                r = params[idx:idx+3]; idx += 3
                t = params[idx:idx+3]; idx += 3
                pose_r[pose] = r
                pose_t[pose] = t

            return cam_rvecs, cam_tvecs, pose_r, pose_t

        # Residuals function
        def residuals(params):
            cam_rvecs, cam_tvecs, pose_r, pose_t = decode_params(params)

            # Precompute rotation matrices
            R_cam = {}
            for ci, r in cam_rvecs.items():
                R_cam[ci], _ = cv2.Rodrigues(r.astype(np.float64))

            R_pose = {}
            for pose, r in pose_r.items():
                R_pose[pose], _ = cv2.Rodrigues(r.astype(np.float64))

            residual_list = []

            for ci, pose, pts2d in measurements:
                cam_id = cameraIds[ci]
                K = state.intrinsics[cam_id]["cameraMatrix"]
                dist = state.intrinsics[cam_id]["distortionCoeffs"]

                # Board pose in reference frame
                r_p = pose_r[pose]
                t_p = pose_t[pose]
                R_p = R_pose[pose]

                if ci == ref_idx:
                    # Reference camera: use board pose directly
                    r_total = r_p
                    t_total = t_p
                else:
                    if ci not in cam_rvecs:
                        # For non-optimized cameras, use static extrinsics
                        extr = state.extrinsics[cam_id]
                        R_c = extr["rotationMatrix"]
                        t_c = extr["translationVector"].reshape(3)
                    else:
                        R_c = R_cam[ci]
                        t_c = cam_tvecs[ci]

                    # Composition: X_cam = R_c * (R_p * X + t_p) + t_c
                    R_total = R_c @ R_p
                    t_total = R_c @ t_p + t_c
                    r_total, _ = cv2.Rodrigues(R_total.astype(np.float64))
                    r_total = r_total.reshape(3)

                # Projection
                proj, _ = cv2.projectPoints(
                    objp,
                    r_total.astype(np.float64),
                    t_total.astype(np.float64),
                    K,
                    dist
                )
                proj = proj.reshape(-1, 2)

                res = (proj - pts2d).reshape(-1)
                residual_list.append(res)

            if not residual_list:
                return np.zeros(0, dtype=np.float64)

            return np.concatenate(residual_list)

        # Optimization (robust, soft_l1)
        result = least_squares(
            residuals,
            x0,
            verbose=2,
            method="trf",       
            loss="soft_l1",     
            f_scale=1.0         
        )

        print(
            f"[BA] Done. Final cost: {result.cost:.4f}, "
            f"RMS per coord: {np.sqrt(2*result.cost/len(result.fun)):.4f} px"
        )

        # Write results back to state.extrinsics
        cam_rvecs_opt, cam_tvecs_opt, pose_r_opt, pose_t_opt = decode_params(result.x)

        # Reference camera remains identity
        state.setExtrinsics(
            refCamId,
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            0.0
        )

        # Store optimized cameras
        for ci in optim_cams:
            cam_id = cameraIds[ci]
            r = cam_rvecs_opt[ci]
            t = cam_tvecs_opt[ci]
            R, _ = cv2.Rodrigues(r.astype(np.float64))
            T = t.reshape(3, 1).astype(np.float64)

            # Approximate RMS for this camera from global residuals
            # (simple average across all measurements)
            cam_residuals = []
            for m_ci, pose, pts2d in measurements:
                if m_ci != ci:
                    continue
                # Reproject with final parameters
                R_p, _ = cv2.Rodrigues(pose_r_opt[pose].astype(np.float64))
                t_p = pose_t_opt[pose].astype(np.float64)

                R_total = R @ R_p
                t_total = R @ t_p + t.flatten()
                r_total, _ = cv2.Rodrigues(R_total)
                K = state.intrinsics[cam_id]["cameraMatrix"]
                dist = state.intrinsics[cam_id]["distortionCoeffs"]

                proj, _ = cv2.projectPoints(
                    objp, r_total, t_total, K, dist
                )
                proj = proj.reshape(-1, 2)
                res = (proj - pts2d).reshape(-1)
                cam_residuals.append(res)

            if cam_residuals:
                cam_residuals = np.concatenate(cam_residuals)
                stereo_rms = float(np.sqrt(np.mean(cam_residuals**2)))
            else:
                stereo_rms = state.extrinsics[cam_id].get("stereoRms", 0.0)

            state.setExtrinsics(cam_id, R, T, stereo_rms)

        return state

    def run(self, imageSet, bundleAdjust=True):
        """Main entry point for calibration pipeline."""
        print("Running intrinsic calibration...")
        state = self.calibrateIntrinsics(imageSet)

        print("\nRunning pairwise extrinsic calibration...")
        state = self.calibrateExtrinsics(imageSet, state, refCamId=imageSet.cameraIds[0])

        if bundleAdjust:
            print("\nRunning global bundle adjustment on extrinsics...")
            state = self.bundleAdjustExtrinsics(imageSet, state, refCamId=imageSet.cameraIds[0])

        return state
