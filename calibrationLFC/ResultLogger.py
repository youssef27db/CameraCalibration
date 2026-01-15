import os
import json
import logging
import numpy as np
from datetime import datetime


class ResultLogger:
    """
    Simple result logger for calibration runs.

    - Writes JSON snapshots into baseDir (timestamped)
    - Writes line-based log file (calibration.log)
    - Manages /data/baseline/baseline.json:
        * if it does not exist -> set it (initial should become baseline first)
        * otherwise -> overwrite only if new health is better
    """

    def __init__(self, baseDir="/data/baseline"):
        self.baseDir = baseDir
        os.makedirs(baseDir, exist_ok=True)

        self.logger = logging.getLogger("calibration")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            log_path = os.path.join(baseDir, "calibration.log")
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(file_handler)

    def _to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        return obj

    def _read_baseline_health(self, baseline_path: str) -> float:
        """
        Reads baseline.json health if available.
        Tries meta.health first; then tries computing via HealthMonitor if missing.
        """
        if not os.path.isfile(baseline_path):
            return -1.0

        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                base = json.load(f)
        except Exception:
            return -1.0

        # 1) direct meta.health
        try:
            h = float(base.get("meta", {}).get("health", -1.0))
            if np.isfinite(h) and h >= 0:
                return h
        except Exception:
            pass

        # 2) fallback: compute (initial or lifcal) based on runType
        try:
            from HealthMonitor import (
                compute_initial_health_from_state,
                compute_lifcal_health_from_combined_dict,
            )
        except Exception:
            return -1.0

        run_type = (base.get("meta", {}) or {}).get("runType", "")
        state = base.get("state", {})

        try:
            if run_type == "initial":
                rep = compute_initial_health_from_state(state)
                return float(rep.get("health", -1.0))
            else:
                # treat as lifcal/combined
                rep = compute_lifcal_health_from_combined_dict(base)
                return float(rep.get("health", -1.0))
        except Exception:
            return -1.0

    def _update_baseline(self, candidate_obj: dict, candidate_score: float, method: str) -> None:
        """
        baseline.json update policy:
        - if baseline.json doesn't exist -> always write (THIS is what you want for first initial run)
        - else write only if candidate_score > old_score
        """
        baseline_path = os.path.join(self.baseDir, "baseline.json")
        old_score = self._read_baseline_health(baseline_path)

        if (not os.path.isfile(baseline_path)) or (candidate_score > old_score):
            try:
                with open(baseline_path, "w", encoding="utf-8") as f:
                    json.dump(candidate_obj, f, indent=2)
                if old_score < 0:
                    self.logger.info(f"Baseline created: baseline.json (new={candidate_score:.2f}, method={method})")
                else:
                    self.logger.info(
                        f"Baseline updated: baseline.json (old={old_score:.2f}, new={candidate_score:.2f}, method={method})"
                    )
            except Exception as e:
                self.logger.error(f"Failed to update baseline.json: {e}")
        else:
            self.logger.info(
                f"Baseline kept: baseline.json (old={old_score:.2f} >= new={candidate_score:.2f}, method={method})"
            )

    def logInitialCalibration(self, calibrationState, meta=None):
        """
        Store initial calibration result as JSON + compute initial health score.
        Also creates baseline.json automatically if it does not exist yet.
        After that: baseline only updated if health improved.
        """
        if meta is None:
            meta = {}

        # extract state
        try:
            state_dict = calibrationState.__getState__()
        except Exception as e:
            self.logger.error(f"Could not extract state dict from CalibrationState: {e}")
            return 0.0

        safe_state = self._to_serializable(state_dict)
        safe_meta = self._to_serializable(meta)

        # compute initial health score from state_dict
        try:
            from HealthMonitor import compute_initial_health_from_state
            report = compute_initial_health_from_state(state_dict)
            score = float(report.get("health", 0.0))
        except Exception as e:
            self.logger.error(f"Initial HealthScore computation failed: {e}", exc_info=True)
            score = 0.0

        # write timestamped json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        imagesetnumber = meta.get("imageDir", "unknown")
        if isinstance(imagesetnumber, str) and "imageset" in imagesetnumber:
            imagesetnumber = imagesetnumber.split("imageset")[-1]

        json_name = f"calibration_initial_imageset{imagesetnumber}_{timestamp}.json"
        json_path = os.path.join(self.baseDir, json_name)

        out = {
            "meta": safe_meta,
            "state": safe_state
        }

        # attach health into meta
        out.setdefault("meta", {})
        out["meta"]["health"] = score
        out["meta"]["method"] = "initial"
        out["meta"]["runType"] = out["meta"].get("runType", "initial")

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write JSON result file '{json_path}': {e}")
            return 0.0

        intr = state_dict.get("intrinsics", {})
        extr = state_dict.get("extrinsics", {})
        self.logger.info(
            f"Stored initial calibration to '{json_name}' "
            f"(intrinsics for {len(intr)} cams, extrinsics for {len(extr)} cams, bundleAdjustf: {meta.get('bundleAdjust', True)})."
        )
        self.logger.info(f"HealthScore = {score:.2f} (method=initial)")

        # baseline policy (this creates baseline on first run!)
        self._update_baseline(out, score, method="initial")

        return score

    def logRecalibration(self, combined_json_path: str, meta=None):
        """
        LiFCal recalibration logger:
        - reads combined.json
        - computes LiFCal HealthScore
        - stores timestamped copy into baseline folder
        - updates baseline.json only if health improved
        """
        if meta is None:
            meta = {}

        # Import here to avoid import errors when not needed
        from HealthMonitor import compute_lifcal_health_from_combined_path

        # 1) read combined json
        try:
            with open(combined_json_path, "r", encoding="utf-8") as f:
                combined = json.load(f)
        except Exception as e:
            self.logger.error(f"Could not read combined.json '{combined_json_path}': {e}")
            return 0.0

        # 2) compute health
        try:
            report = compute_lifcal_health_from_combined_path(combined_json_path)
            score = float(report.get("health", 0.0))
        except Exception as e:
            self.logger.error(f"LiFCal HealthScore computation failed: {e}", exc_info=True)
            score = 0.0

        # 3) store timestamped copy
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_name = f"calibration_lifcal_{timestamp}.json"
        out_path = os.path.join(self.baseDir, json_name)

        combined.setdefault("meta", {})
        combined["meta"].update(self._to_serializable(meta))
        combined["meta"]["health"] = score
        combined["meta"]["method"] = "lifcal"
        combined["meta"]["runType"] = combined["meta"].get("runType", "recalib")

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(combined, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write LiFCal result '{out_path}': {e}")
            return 0.0

        num_cams = len(combined.get("state", {}).get("parameters", {}))
        self.logger.info(f"Stored lifcal recalibration to '{json_name}' (cams={num_cams}).")
        self.logger.info(f"HealthScore = {score:.2f} (method=lifcal)")

        # 4) baseline update only if better
        self._update_baseline(combined, score, method="lifcal")

        return score
