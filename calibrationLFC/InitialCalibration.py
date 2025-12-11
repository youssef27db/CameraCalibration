import cv2
import numpy as np
from CalibrationState import CalibrationState
from scipy.optimize import least_squares


class InitialCalibration:

    def __init__(self, chessboardSize=(8,8)):
        self.chessboardSize = chessboardSize
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # ---------------------------
    # Helper: object points, corner detection
    # ---------------------------
    def createObjectPoints(self):
        cols, rows = self.chessboardSize
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
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
    # Step 3: Global Bundle Adjustment auf Extrinsics (robust)
    # -------------------------------------------------------
    def bundleAdjustExtrinsics(self, imageSet, state, refCamId):
        """
        Globales Bundle Adjustment:
        - Intrinsics bleiben fix (aus state.intrinsics)
        - Optimiert werden:
          * Extrinsics jeder Nicht-Referenzkamera (R_cam, t_cam)
          * Pose des Schachbretts pro Pose (R_board_p, t_board_p) im
            Referenz-Kamerakoordinatensystem.
        """
        objp = self.createObjectPoints()
        cameraIds = list(imageSet.cameraIds)
        if refCamId not in cameraIds:
            raise ValueError("refCamId not in imageSet.cameraIds")

        ref_idx = cameraIds.index(refCamId)

        # ---------- 1) Initiale Board-Posen via PnP-RANSAC in der Referenzkamera ----------
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

            # solvePnPRansac für robustere Board-Posen
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
                # zu wenig verlässliche Punkte -> Pose skippen
                continue

            pose_rvecs[pose] = rvec.astype(np.float64).reshape(3)
            pose_tvecs[pose] = tvec.astype(np.float64).reshape(3)
            valid_poses.append(pose)

        if len(valid_poses) < 3:
            print("[BA] Not enough valid poses for bundle adjustment.")
            return state

        # ---------- 2) Extrinsic-Initialisierung aus state.extrinsics ----------
        # Wir optimieren nur Nicht-Referenzkameras
        optim_cams = []
        cam_param_index = {}  # global_cam_idx -> 0..(num_optim_cams-1)

        for ci, cam in enumerate(cameraIds):
            if cam == refCamId:
                continue
            if cam not in state.extrinsics:
                # falls stereoCalibrate die Kamera nicht gesetzt hat
                continue

            optim_cams.append(ci)
            cam_param_index[ci] = len(cam_param_index)

        num_optim_cams = len(optim_cams)
        if num_optim_cams == 0:
            print("[BA] No extrinsic data to optimize, skipping BA.")
            return state

        # ---------- 3) Messungen sammeln (alle Kameras, alle gültigen Posen) ----------
        # Jede Messung: (cam_idx, pose, 2D-Punkte)
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

        # ---------- 4) Parametervektor initialisieren ----------
        # Layout:
        # [ cams (ohne ref): (r_cam(3), t_cam(3)) ... , poses: (r_p(3), t_p(3)) ... ]

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

        # ---------- Hilfsfunktionen für Parameter-Decoding ----------
        def decode_params(params):
            # cams
            cam_rvecs = {}
            cam_tvecs = {}
            idx = 0
            for ci in optim_cams:
                r = params[idx:idx+3]; idx += 3
                t = params[idx:idx+3]; idx += 3
                cam_rvecs[ci] = r
                cam_tvecs[ci] = t

            # poses
            pose_r = {}
            pose_t = {}
            for pose in valid_poses:
                r = params[idx:idx+3]; idx += 3
                t = params[idx:idx+3]; idx += 3
                pose_r[pose] = r
                pose_t[pose] = t

            return cam_rvecs, cam_tvecs, pose_r, pose_t

        # ---------- 5) Residuen-Funktion ----------
        def residuals(params):
            cam_rvecs, cam_tvecs, pose_r, pose_t = decode_params(params)

            # Precompute Rotationsmatrizen
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

                # Pose des Brettes im Ref-System
                r_p = pose_r[pose]
                t_p = pose_t[pose]
                R_p = R_pose[pose]

                if ci == ref_idx:
                    # Referenzkamera: direkt Board-Pose
                    r_total = r_p
                    t_total = t_p
                else:
                    if ci not in cam_rvecs:
                        # für Kameras, die nicht optimiert werden, benutzen wir
                        # ihre alten Extrinsics (statisch)
                        extr = state.extrinsics[cam_id]
                        R_c = extr["rotationMatrix"]
                        t_c = extr["translationVector"].reshape(3)
                    else:
                        R_c = R_cam[ci]
                        t_c = cam_tvecs[ci]

                    # Komposition: X_cam = R_c * (R_p * X + t_p) + t_c
                    R_total = R_c @ R_p
                    t_total = R_c @ t_p + t_c
                    r_total, _ = cv2.Rodrigues(R_total.astype(np.float64))
                    r_total = r_total.reshape(3)

                # Projektion
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

        # ---------- 6) Optimierung (robust, soft_l1) ----------
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

        # ---------- 7) Ergebnis zurück in state.extrinsics schreiben ----------
        cam_rvecs_opt, cam_tvecs_opt, pose_r_opt, pose_t_opt = decode_params(result.x)

        # Referenzkamera bleibt Identität
        state.setExtrinsics(
            refCamId,
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            0.0
        )

        # Optimierte Kameras
        for ci in optim_cams:
            cam_id = cameraIds[ci]
            r = cam_rvecs_opt[ci]
            t = cam_tvecs_opt[ci]
            R, _ = cv2.Rodrigues(r.astype(np.float64))
            T = t.reshape(3, 1).astype(np.float64)

            # Approx. RMS dieser Kamera aus globalen Residuen
            # (einfacher Durchschnitt über alle Messungen der Kamera)
            cam_residuals = []
            for m_ci, pose, pts2d in measurements:
                if m_ci != ci:
                    continue
                # nochmal kurz projizieren mit finalen Parametern
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

    # -------------------------------------------------------
    # PUBLIC API: run(imageSet)
    # -------------------------------------------------------
    def run(self, imageSet, bundleAdjust=True):
        """
        Main entry point.
        Returns a filled CalibrationState.
        """
        print("Running intrinsic calibration...")
        state = self.calibrateIntrinsics(imageSet)

        print("\nRunning pairwise extrinsic calibration...")
        state = self.calibrateExtrinsics(imageSet, state, refCamId=imageSet.cameraIds[0])

        if bundleAdjust:
            # Globales Bundle Adjustment (optional, aber empfohlen)
            print("\nRunning global bundle adjustment on extrinsics...")
            state = self.bundleAdjustExtrinsics(imageSet, state, refCamId=imageSet.cameraIds[0])

        return state
