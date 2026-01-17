import os
import cv2
import numpy as np
from typing import Optional, Dict


def build_matchers():
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,  # must be divisible by 16
        blockSize=7,
        P1=8 * 7 * 7,
        P2=32 * 7 * 7,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
        disp12MaxDiff=1,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = None
    wls = None

    try:
        if hasattr(cv2, "ximgproc"):
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls.setLambda(6000)
            wls.setSigmaColor(1.0)
    except Exception:
        right_matcher = None
        wls = None

    return left_matcher, right_matcher, wls


_LEFT_MATCHER, _RIGHT_MATCHER, _WLS = build_matchers()


def imwrite_u16(path: str, img_u16: np.ndarray) -> bool:
    """Write uint16 png, create dirs automatically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img_u16.dtype != np.uint16:
        img_u16 = img_u16.astype(np.uint16)
    return bool(cv2.imwrite(path, img_u16))


def disparity(left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
    """
    Returns disparity as float32 with NaNs for invalid (<=0).
    """
    dl = _LEFT_MATCHER.compute(left_gray, right_gray)  # int16 scaled by 16

    if _RIGHT_MATCHER is not None and _WLS is not None:
        dr = _RIGHT_MATCHER.compute(right_gray, left_gray)
        d = _WLS.filter(dl, left_gray, None, dr)
    else:
        d = dl

    d = d.astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    return d


def to_u16_from_disp(disp: np.ndarray) -> np.ndarray:
    """
    Disp -> uint16 scaling robust via percentile.
    """
    disp = np.nan_to_num(disp, nan=0.0, posinf=0.0, neginf=0.0)

    mx = float(np.percentile(disp, 99))
    if not np.isfinite(mx) or mx <= 1e-6:
        return np.zeros(disp.shape, dtype=np.uint16)

    disp = np.clip(disp, 0.0, mx)
    depth16 = (disp / mx * 65535.0).astype(np.uint16)
    return depth16


def fuse_disparities_median(disp_list: list) -> np.ndarray:
    """
    Fuses disparities robustly without 'All-NaN slice' warnings.
    Returns float32 disparity with 0 where nothing is valid.
    """
    stack = np.stack(disp_list, axis=0)  # (K,H,W)

    # Locations where at least one value is valid
    valid_any = np.any(~np.isnan(stack), axis=0)

    # Initialize result with zeros
    disp_final = np.zeros(stack.shape[1:], dtype=np.float32)

    # Compute median only where we have valid data
    if np.any(valid_any):
        disp_final[valid_any] = np.nanmedian(stack[:, valid_any], axis=0)

    # Remove non-positive disparities
    disp_final[disp_final <= 0] = 0.0
    return disp_final


def generate_depthmaps_for_pose(cam_to_focus_path: Dict[str, str]) -> Dict[str, np.ndarray]:
    """
    cam_to_focus_path: {"Center": "/.../pose000_Center.png", "Up1": "...", ...}
    returns: {"Center": depth_u16, "Up1": depth_u16, ...}
    """

    # 1) Load images
    imgs = {}
    for cam, path in cam_to_focus_path.items():
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        imgs[cam] = im

    if len(imgs) < 2:
        return {}

    # helper
    def opposite_name(name: str) -> Optional[str]:
        if name.startswith("Left"):
            return name.replace("Left", "Right", 1)
        if name.startswith("Right"):
            return name.replace("Right", "Left", 1)
        if name.startswith("Up"):
            return name.replace("Up", "Down", 1)
        if name.startswith("Down"):
            return name.replace("Down", "Up", 1)
        return None

    depthmaps = {}

    for cam, img in imgs.items():
        disp_list = []

        if cam == "Center":
            # Center uses all other views
            for other_cam, other_img in imgs.items():
                if other_cam == "Center":
                    continue
                d = disparity(img, other_img)
                # Keep only if there is any valid disparity
                if np.any(~np.isnan(d)):
                    disp_list.append(d)

        else:
            # Opposite direction if available
            opp = opposite_name(cam)
            if opp is not None and opp in imgs:
                if cam.startswith("Right") or cam.startswith("Down"):
                    d = disparity(imgs[opp], img)
                else:
                    d = disparity(img, imgs[opp])

                if np.any(~np.isnan(d)):
                    disp_list.append(d)

            # Add Center view
            if "Center" in imgs:
                d = disparity(img, imgs["Center"])
                if np.any(~np.isnan(d)):
                    disp_list.append(d)

        if not disp_list:
            # No valid disparity found; skip
            continue

        # Robust fusion
        disp_final = fuse_disparities_median(disp_list)

        # Convert to uint16 depth map
        depth16 = to_u16_from_disp(disp_final)
        depthmaps[cam] = depth16

    return depthmaps
