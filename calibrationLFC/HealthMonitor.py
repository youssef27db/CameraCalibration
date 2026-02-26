"""
HealthScore Formulas (0..100)

(A) INITIAL Calibration (OpenCV/Controller JSON)
    HealthScore_initial = 100 * [ 1 - (0.6 * E_reproj^2 + 0.4 * E_rms^2) ]

    E_reproj = clamp( (reproj_mean - R_BEST) / (R_WORST - R_BEST), 0, 1 )
    E_rms    = clamp( (rms_mean    - S_BEST) / (S_WORST - S_BEST), 0, 1 )

    Uses:
      - intrinsics[*].reprojectionError  (per camera)
      - extrinsics[*].stereoRms          (per camera)

(B) LiFCal Calibration (protocol-based)
    HealthScore_lifcal = 100 * [ 1 - (0.6 * E_std^2 + 0.4 * E_mae^2) ]

    E_std = clamp( ( sqrt(x_std^2 + y_std^2) - S_BEST ) / (S_WORST - S_BEST), 0, 1 )
    E_mae = clamp( ( sqrt(x_mae^2 + y_mae^2) - R_BEST ) / (R_WORST - R_BEST), 0, 1 )

    Uses:
      - std.Dev. x/y  and mae x/y from LiFCal calibrationProtocol.txt

Shared thresholds (for BOTH methods):
  R_BEST = 0.03, R_WORST = 1.0, S_BEST = 0.0, S_WORST = 50.0
"""

import re
import json
import math
from typing import Dict, Any, Optional

"""
@brief HealthMonitor module for computing calibration health scores.

"""

# Shared thresholds
R_BEST = 0.03
R_WORST = 1.0
S_BEST = 0.0
S_WORST = 50.0


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    @brief Clamp value between lower and upper bounds.
    
    @param x Value to clamp
    @param lo Lower bound (default: 0.0)
    @param hi Upper bound (default: 1.0)
    @return Clamped value
    """
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def safe_float(v, default=None):
    """
    @brief Convert value to float safely, return default on error.
    
    @param v Value to convert
    @param default Default value to return on conversion error
    @return Converted float value or default
    """
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def score_from_errors(E_a: float, E_b: float, w_a: float = 0.6, w_b: float = 0.4) -> float:
    """
    @brief Compute health score from two normalized error metrics.
    
    @param E_a First normalized error metric (0..1)
    @param E_b Second normalized error metric (0..1)
    @param w_a Weight for first error metric (default: 0.6)
    @param w_b Weight for second error metric (default: 0.4)
    @return Health score between 0 and 100
    """
    val = 100.0 * (1.0 - (w_a * (E_a ** 2) + w_b * (E_b ** 2)))
    return clamp(val / 100.0, 0.0, 1.0) * 100.0


def compute_initial_health_from_state(
    state_dict: Dict[str, Any],
    r_best: float = R_BEST,
    r_worst: float = R_WORST,
    s_best: float = S_BEST,
    s_worst: float = S_WORST,
) -> Dict[str, Any]:
    """
    @brief Compute health score from initial calibration state.

    @param state_dict Dictionary containing calibration state with intrinsics and extrinsics
    @param r_best Best reprojection error threshold (default: R_BEST)
    @param r_worst Worst reprojection error threshold (default: R_WORST)
    @param s_best Best stereo RMS threshold (default: S_BEST)
    @param s_worst Worst stereo RMS threshold (default: S_WORST)
    @return Dictionary with health score, method name, and detailed metrics
    """
    st = state_dict.get("state", state_dict)
    intr = st.get("intrinsics", {}) or {}
    extr = st.get("extrinsics", {}) or {}

    reproj_vals = []
    rms_vals = []

    for cam, intr_data in intr.items():
        reproj = safe_float(intr_data.get("reprojectionError"), None)
        if reproj is not None:
            reproj_vals.append(reproj)

        ex = extr.get(cam, {})
        rms = safe_float(ex.get("stereoRms"), None)
        if rms is not None:
            rms_vals.append(rms)

    if not reproj_vals and not rms_vals:
        return {
            "health": 0.0,
            "method": "initial",
            "details": {"reason": "no reprojectionError and no stereoRms found"},
        }

    reproj_mean = (sum(reproj_vals) / len(reproj_vals)) if reproj_vals else r_worst
    rms_mean = (sum(rms_vals) / len(rms_vals)) if rms_vals else s_worst

    E_reproj = clamp((reproj_mean - r_best) / (r_worst - r_best), 0.0, 1.0)
    E_rms = clamp((rms_mean - s_best) / (s_worst - s_best), 0.0, 1.0)

    score = score_from_errors(E_reproj, E_rms)

    return {
        "health": float(score),
        "method": "initial",
        "details": {
            "reproj_mean": float(reproj_mean),
            "rms_mean": float(rms_mean),
            "E_reproj": float(E_reproj),
            "E_rms": float(E_rms),
            "thresholds": {
                "R_BEST": float(r_best),
                "R_WORST": float(r_worst),
                "S_BEST": float(s_best),
                "S_WORST": float(s_worst),
            },
        },
    }


def parse_lifcal_protocol(protocol_path: str) -> Dict[str, Optional[float]]:
    """
    @brief Parse LiFCal calibration protocol text file for error metrics.
    
    @param protocol_path Path to the calibration protocol text file
    @return Dictionary with x_std, y_std, x_mae, y_mae values or None if not found
    """
    txt = open(protocol_path, "r", encoding="utf-8", errors="ignore").read()

    def find_float(pattern: str) -> Optional[float]:
        m = re.search(pattern, txt, re.MULTILINE)
        if not m:
            return None
        return safe_float(m.group(1), None)

    x_std = find_float(r"std\.\s*Dev\.\s*x:\s*([0-9.+-eE]+)")
    y_std = find_float(r"std\.\s*Dev\.\s*y:\s*([0-9.+-eE]+)")
    x_mae = find_float(r"mae\s*x:\s*([0-9.+-eE]+)")
    y_mae = find_float(r"mae\s*y:\s*([0-9.+-eE]+)")

    return {"x_std": x_std, "y_std": y_std, "x_mae": x_mae, "y_mae": y_mae}


def compute_lifcal_health_from_xy(
    x_std: float,
    y_std: float,
    x_mae: float,
    y_mae: float,
    r_best: float = R_BEST,
    r_worst: float = R_WORST,
    s_best: float = S_BEST,
    s_worst: float = S_WORST,
) -> Dict[str, Any]:
    """
    @brief Compute LiFCal health score from x/y standard deviation and MAE values.
    
    @param x_std Standard deviation in x direction
    @param y_std Standard deviation in y direction
    @param x_mae Mean absolute error in x direction
    @param y_mae Mean absolute error in y direction
    @param r_best Best reprojection error threshold (default: R_BEST)
    @param r_worst Worst reprojection error threshold (default: R_WORST)
    @param s_best Best stereo RMS threshold (default: S_BEST)
    @param s_worst Worst stereo RMS threshold (default: S_WORST)
    @return Dictionary with health score, method name, and detailed metrics
    """
    std_norm = (x_std * x_std + y_std * y_std) ** 0.5
    mae_norm = (x_mae * x_mae + y_mae * y_mae) ** 0.5

    E_std = clamp((std_norm - s_best) / (s_worst - s_best), 0.0, 1.0)
    E_mae = clamp((mae_norm - r_best) / (r_worst - r_best), 0.0, 1.0)

    score = score_from_errors(E_std, E_mae)

    return {
        "health": float(score),
        "method": "lifcal",
        "details": {
            "std_norm": float(std_norm),
            "mae_norm": float(mae_norm),
            "E_std": float(E_std),
            "E_mae": float(E_mae),
            "thresholds": {
                "R_BEST": float(r_best),
                "R_WORST": float(r_worst),
                "S_BEST": float(s_best),
                "S_WORST": float(s_worst),
            },
        },
    }

def compute_lifcal_health_from_combined_dict(combined: Dict[str, Any]) -> Dict[str, Any]:
    """
    @brief Compute LiFCal health from combined.json structure.

    @param combined Dictionary from combined.json with parameters for all cameras
    @return Dictionary with global health score, method name, and per-camera metrics
    """
    params_all = combined.get("state", {}).get("parameters", {})
    if not isinstance(params_all, dict) or len(params_all) == 0:
        return {"health": 0.0, "method": "lifcal", "perCam": {}}

    per_cam = {}
    scores = []

    for cam, p in params_all.items():
        rep = (p or {}).get("reprojection", {}) or {}

        x_std = rep.get("x_std", None)
        y_std = rep.get("y_std", None)
        x_mae = rep.get("x_mae", None)
        y_mae = rep.get("y_mae", None)

        # Skip camera if any field is missing
        if any(v is None for v in [x_std, y_std, x_mae, y_mae]):
            per_cam[cam] = {"health": 0.0, "reason": "missing reprojection fields"}
            continue

        std_mag = math.sqrt(float(x_std) ** 2 + float(y_std) ** 2)
        mae_mag = math.sqrt(float(x_mae) ** 2 + float(y_mae) ** 2)

        E_std = clamp((std_mag - S_BEST) / (S_WORST - S_BEST), 0.0, 1.0)
        E_mae = clamp((mae_mag - R_BEST) / (R_WORST - R_BEST), 0.0, 1.0)

        health = 100.0 * (1.0 - (0.6 * (E_std ** 2) + 0.4 * (E_mae ** 2)))
        health = clamp(health, 0.0, 100.0)

        per_cam[cam] = {
            "health": float(health),
            "std_mag": float(std_mag),
            "mae_mag": float(mae_mag),
            "E_std": float(E_std),
            "E_mae": float(E_mae),
        }
        scores.append(float(health))

    if len(scores) == 0:
        return {"health": 0.0, "method": "lifcal", "perCam": per_cam}

    # Global health: average across all cameras
    global_health = sum(scores) / len(scores)

    return {"health": float(global_health), "method": "lifcal", "perCam": per_cam}


def compute_lifcal_health_from_combined_path(combined_json_path: str) -> Dict[str, Any]:
    """
    @brief Load combined.json from file and compute LiFCal health.
    
    @param combined_json_path Path to the combined.json file
    @return Dictionary with global health score, method name, and per-camera metrics
    """
    with open(combined_json_path, "r", encoding="utf-8") as f:
        combined = json.load(f)
    return compute_lifcal_health_from_combined_dict(combined)