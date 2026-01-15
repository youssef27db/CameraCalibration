import os
import json
import logging
import numpy as np
from datetime import datetime


class ResultLogger:
    """
    Simple result logger for calibration runs.

    - Writes a JSON snapshot of CalibrationState.__getState__()
    - Writes a line-based log file (calibration.log)
    """

    def __init__(self, baseDir="/data/baseline"):
        self.baseDir = baseDir
        os.makedirs(baseDir, exist_ok=True)

        # Set up Python logging
        self.logger = logging.getLogger("calibration")
        self.logger.setLevel(logging.INFO)

        # Avoid adding handlers twice if multiple ResultLogger instances are created
        if not self.logger.handlers:
            log_path = os.path.join(baseDir, "calibration.log")
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(file_handler)

    def _to_serializable(self, obj):
        # NumPy Arrays -> Listen
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # NumPy-Skalare -> normale float/int
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        # dict rekursiv
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        # Liste/Tuple rekursiv
        if isinstance(obj, (list, tuple)):
            return [self._to_serializable(v) for v in obj]
        # alles andere so lassen (str, float, int, None, bool, ...)
        return obj

    def logInitialCalibration(self, calibrationState, meta=None):
        """
        Store the full calibration state as JSON and write a short summary to the log.

        """
        if meta is None:
            meta = {}

        # Try to get a dictionary representation from CalibrationState
        try:
            state_dict = calibrationState.__getState__()
        except Exception as e:
            self.logger.error(f"Could not extract state dict from CalibrationState: {e}")
            return

        safe_state = self._to_serializable(state_dict)
        safe_meta  = self._to_serializable(meta)

        # File name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        imagesetnumber = meta.get("imageDir", "unknown").split("imageset")[-1]
        json_name = f"calibration_initial_imageset{imagesetnumber}_{timestamp}.json"
        json_path = os.path.join(self.baseDir, json_name)

        # Wrap everything into one JSON object
        out = {
            "meta": safe_meta,
            "state": safe_state
        }

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to write JSON result file '{json_path}': {e}")
            return

        # Short textual summary
        intr = state_dict.get("intrinsics", {})
        extr = state_dict.get("extrinsics", {})

        num_cams_intr = len(intr)
        num_cams_extr = len(extr)
        run_type = meta.get("runType", "initial")
        bundle_adjust = meta.get("bundleAdjust", True)

        self.logger.info(
            f"Stored {run_type} calibration to '{json_name}' "
            f"(intrinsics for {num_cams_intr} cams, extrinsics for {num_cams_extr} cams, bundleAdjustf: {bundle_adjust})."
        )
    

    def logRecalibration(self, calibrationState, meta=None):
        pass
