# ============================================================
# HealthScore-Formeln (0..100)
# ============================================================
#
# (A) INITIAL Calibration (OpenCV/Controller JSON)
# -----------------------------------------------
# HealthScore_initial = 100 * [ 1 - (0.6 * E_reproj^2 + 0.4 * E_rms^2) ]
#
# E_reproj = clamp( (reproj_mean - R_BEST) / (R_WORST - R_BEST), 0, 1 )
# E_rms    = clamp( (rms_mean    - S_BEST) / (S_WORST - S_BEST), 0, 1 )
#
# Uses:
#   - intrinsics[*].reprojectionError  (per camera)
#   - extrinsics[*].stereoRms          (per camera)
#
#
# (B) LiFCal Calibration (protocol-based)
# ---------------------------------------
# HealthScore_lifcal = 100 * [ 1 - (0.6 * E_std^2 + 0.4 * E_mae^2) ]
#
# E_std = clamp( ( sqrt(x_std^2 + y_std^2) - S_BEST ) / (S_WORST - S_BEST), 0, 1 )
# E_mae = clamp( ( sqrt(x_mae^2 + y_mae^2) - R_BEST ) / (R_WORST - R_BEST), 0, 1 )
#
# Uses:
#   - std.Dev. x/y  and mae x/y from LiFCal calibrationProtocol.txt
#
# Shared thresholds (for BOTH methods):
#   R_BEST = 0.03, R_WORST = 1.0, S_BEST = 0.0, S_WORST = 50.0
# ============================================================

import re
import json
from typing import Dict, Any, Optional


# -------------------------
# Shared thresholds
# -------------------------
R_BEST: float = 0.03
R_WORST: float = 1.0
S_BEST: float = 0.0
S_WORST: float = 50.0


# -------------------------
# Utilities
# -------------------------
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _score_from_errors(E_a: float, E_b: float, w_a: float = 0.6, w_b: float = 0.4) -> float:
    # 100 * (1 - (w_a*E_a^2 + w_b*E_b^2)), clamped
    val = 100.0 * (1.0 - (w_a * (E_a ** 2) + w_b * (E_b ** 2)))
    return clamp(val / 100.0, 0.0, 1.0) * 100.0


# ============================================================
# INITIAL calibration scoring
# ============================================================
def compute_initial_health_from_state(
    state_dict: Dict[str, Any],
    r_best: float = R_BEST,
    r_worst: float = R_WORST,
    s_best: float = S_BEST,
    s_worst: float = S_WORST,
) -> Dict[str, Any]:
    """
    Controller-friendly: returns dict with key 'health'.

    Reads from state_dict:
      state.intrinsics[cam].reprojectionError
      state.extrinsics[cam].stereoRms

    Returns:
      {"health": float, "method": "initial", "details": {...}}
    """
    st = state_dict.get("state", state_dict)  # allow passing just "state"
    intr = st.get("intrinsics", {}) or {}
    extr = st.get("extrinsics", {}) or {}

    reproj_vals = []
    rms_vals = []

    for cam, intr_data in intr.items():
        reproj = _safe_float(intr_data.get("reprojectionError"), None)
        if reproj is not None:
            reproj_vals.append(reproj)

        ex = extr.get(cam, {})
        rms = _safe_float(ex.get("stereoRms"), None)
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

    score = _score_from_errors(E_reproj, E_rms)

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


# ============================================================
# LiFCal scoring (protocol-based)
# ============================================================
def parse_lifcal_protocol(protocol_path: str) -> Dict[str, Optional[float]]:
    txt = open(protocol_path, "r", encoding="utf-8", errors="ignore").read()

    def find_float(pattern: str) -> Optional[float]:
        m = re.search(pattern, txt, re.MULTILINE)
        if not m:
            return None
        return _safe_float(m.group(1), None)

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
    std_norm = (x_std * x_std + y_std * y_std) ** 0.5
    mae_norm = (x_mae * x_mae + y_mae * y_mae) ** 0.5

    E_std = clamp((std_norm - s_best) / (s_worst - s_best), 0.0, 1.0)
    E_mae = clamp((mae_norm - r_best) / (r_worst - r_best), 0.0, 1.0)

    score = _score_from_errors(E_std, E_mae)

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


def compute_lifcal_health_from_protocol(protocol_path: str) -> Dict[str, Any]:
    vals = parse_lifcal_protocol(protocol_path)
    if any(v is None for v in vals.values()):
        return {"health": 0.0, "method": "lifcal", "details": {"reason": "missing std/mae values"}}

    return compute_lifcal_health_from_xy(
        vals["x_std"], vals["y_std"], vals["x_mae"], vals["y_mae"]
    )


def compute_lifcal_health_from_parameters_json(parameters_json_path: str) -> Dict[str, Any]:
    """
    If you store LiFCal numbers in parameters.json somewhere, you can use this too.
    Expects e.g.:
      state.lifcal.reprojection.x_std, ...
    """
    with open(parameters_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    st = data.get("state", {})
    rep = None

    if isinstance(st.get("lifcal"), dict):
        rep = st["lifcal"].get("reprojection")

    if rep is None and isinstance(st.get("reprojection"), dict):
        rep = st.get("reprojection")

    if not isinstance(rep, dict):
        return {"health": 0.0, "method": "lifcal", "details": {"reason": "no reprojection dict found"}}

    x_std = _safe_float(rep.get("x_std"), None)
    y_std = _safe_float(rep.get("y_std"), None)
    x_mae = _safe_float(rep.get("x_mae"), None)
    y_mae = _safe_float(rep.get("y_mae"), None)

    if None in (x_std, y_std, x_mae, y_mae):
        return {"health": 0.0, "method": "lifcal", "details": {"reason": "missing x/y std/mae"}}

    return compute_lifcal_health_from_xy(x_std, y_std, x_mae, y_mae)