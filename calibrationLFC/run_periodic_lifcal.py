import os
import sys
import time
import json
import logging
import subprocess
from datetime import datetime

""" 
@brief Periodic LiFCal scheduler that prepares imagesets, runs calibration, and logs health.

"""

INTERVAL_SECONDS = 5 * 60  # 5 Minutes interval

RUN_IMAGESET_CREATION = "/data/external/LiFCal/LiFCal_Data/run_imageset_creation.py"
RUN_LIFCAL_CALIB      = "/data/external/LiFCal/LiFCal_Data/run_LiFCal_calibration.py"

LIFCAL_OUT_ROOT = "/data/external/LiFCal/LiFCal_Data/Recalibration/CalibrationByCamera"
BASELINE_DIR    = "/data/baseline"
LOG_PATH        = os.path.join(BASELINE_DIR, "calibration.log")

LOCK_PATH = "/tmp/lifcal_periodic.lock"

# Prepared images
ROOT_IMAGESET = "/data/external/LiFCal/LiFCal_Data/Recalibration/LiFCal_Imageset"
FOCUS_ROOT    = os.path.join(ROOT_IMAGESET, "focus")   # focus/<CamId>/*.png
DEPTH_ROOT    = os.path.join(ROOT_IMAGESET, "depth")   # depth/<CamId>/*_depth.png

CAMERA_IDS = [
    "Center", "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3"
]

# Minimum number of poses required per camera
MIN_POSES_PER_CAM = 1


def setup_logger():
    """
    @brief Create logger that writes to file and stdout.
    
    @return Logger instance configured for calibration logging
    """
    os.makedirs(BASELINE_DIR, exist_ok=True)
    logger = logging.getLogger("lifcal_periodic")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(sh)

    return logger


def acquire_lock(logger) -> bool:
    """
    @brief Create lock file to prevent overlapping scheduler cycles.
    
    @param logger Logger instance for status messages
    @return True if lock was acquired, False otherwise
    """
    if os.path.exists(LOCK_PATH):
        logger.warning(f"Lock exists, skipping this cycle: {LOCK_PATH}")
        return False
    try:
        with open(LOCK_PATH, "w") as f:
            f.write(str(os.getpid()))
        return True
    except Exception as e:
        logger.error(f"Could not create lock: {e}")
        return False


def release_lock(logger):
    """
    @brief Remove lock file after a cycle completes.
    
    @param logger Logger instance for error messages
    """
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except Exception as e:
        logger.error(f"Could not remove lock: {e}")


def run_cmd(logger, cmd, cwd=None) -> int:
    """
    @brief Run a subprocess, log combined stdout/stderr, and return exit code.
    
    @param logger Logger instance for command output
    @param cmd Command as list of strings
    @param cwd Working directory for the command (default: None)
    @return Exit code of the subprocess
    """
    logger.info(f"RUN: {' '.join(cmd)}")
    try:
        p = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        out = p.stdout or ""
        if len(out) > 5000:
            logger.info(out[:5000] + "\n... (truncated) ...")
        else:
            logger.info(out)
        return int(p.returncode)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        return 1


def count_images_in_dir(d: str, exts=(".png", ".jpg", ".jpeg")) -> int:
    """
    @brief Count images in directory matching expected extensions.
    
    @param d Directory path to scan
    @param exts Tuple of file extensions to match (default: (".png", ".jpg", ".jpeg"))
    @return Number of matching image files
    """
    if not os.path.isdir(d):
        return 0
    cnt = 0
    for fn in os.listdir(d):
        if fn.lower().endswith(exts):
            cnt += 1
    return cnt


def needs_imageset_creation(logger) -> bool:
    """
    @brief Check if imageset exists and has enough images per camera; trigger creation if not.
    
    @param logger Logger instance for status messages
    @return True if imageset needs to be created, False if it already exists
    """
    if not os.path.isdir(ROOT_IMAGESET):
        logger.info("Imageset missing: ROOT_IMAGESET not found.")
        return True
    if not os.path.isdir(FOCUS_ROOT) or not os.path.isdir(DEPTH_ROOT):
        logger.info("Imageset missing: focus/depth folders not found.")
        return True

    for cam in CAMERA_IDS:
        fdir = os.path.join(FOCUS_ROOT, cam)
        ddir = os.path.join(DEPTH_ROOT, cam)

        fcnt = count_images_in_dir(fdir, exts=(".png", ".jpg", ".jpeg"))
        dcnt = count_images_in_dir(ddir, exts=(".png", ".jpg", ".jpeg"))

        if fcnt < MIN_POSES_PER_CAM:
            logger.info(f"Imageset not ready: focus/{cam} has {fcnt} images (<{MIN_POSES_PER_CAM}).")
            return True
        if dcnt < MIN_POSES_PER_CAM:
            logger.info(f"Imageset not ready: depth/{cam} has {dcnt} images (<{MIN_POSES_PER_CAM}).")
            return True
    logger.info("Imageset looks ready. Skipping run_imageset_creation.py.")
    return False


def find_latest_run_dir(out_root: str) -> str:
    """
    @brief Return latest run_* directory inside output root, or empty string if none.
    
    @param out_root Root directory containing run_* folders
    @return Path to the latest run directory, or empty string
    """
    if not os.path.isdir(out_root):
        return ""
    runs = []
    for name in os.listdir(out_root):
        p = os.path.join(out_root, name)
        if os.path.isdir(p) and name.startswith("run_"):
            runs.append(p)
    if not runs:
        return ""
    runs.sort(key=lambda p: os.path.getmtime(p))
    return runs[-1]


def combined_json_path(run_dir: str) -> str:
    """
    @brief Return path to combined.json inside run directory if present.
    
    @param run_dir Path to run directory
    @return Path to combined.json file, or empty string if not found
    """
    if not run_dir:
        return ""
    p = os.path.join(run_dir, "combined.json")
    return p if os.path.isfile(p) else ""


def try_log_lifcal_with_resultlogger(logger, combined_path: str):
    """
    @brief Use ResultLogger to record LiFCal health score and update baseline.
    
    @param logger Logger instance for status messages
    @param combined_path Path to combined.json from LiFCal
    @return True if logging succeeded, False otherwise
    """
    try:
        repo_dir = "/data/calibrationLFC"
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)

        from ResultLogger import ResultLogger
        rl = ResultLogger(baseDir=BASELINE_DIR)

        meta = {
            "runType": "recalib",
            "source": "periodic_scheduler",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        score = rl.logRecalibration(combined_path, meta=meta)
        logger.info(f"LiFCal HealthScore logged via ResultLogger: {score:.2f}")
        return True
    except Exception as e:
        logger.warning(f"Could not use ResultLogger for LiFCal logging: {e}")
        return False


def run_cycle(logger):
    """
    @brief Execute one scheduler cycle: prepare imageset, run LiFCal, log health.
    
    @param logger Logger instance for cycle logging
    """
    # 1) Only prepare imageset if needed
    if needs_imageset_creation(logger):
        if not os.path.isfile(RUN_IMAGESET_CREATION):
            logger.error(f"Missing: {RUN_IMAGESET_CREATION}")
            return

        rc = run_cmd(logger, [sys.executable, RUN_IMAGESET_CREATION])
        if rc != 0:
            logger.error(f"run_imageset_creation failed (rc={rc}). Skipping LiFCal.")
            return

    # 2) LiFCal calibration
    if not os.path.isfile(RUN_LIFCAL_CALIB):
        logger.error(f"Missing: {RUN_LIFCAL_CALIB}")
        return

    rc = run_cmd(logger, [sys.executable, RUN_LIFCAL_CALIB])
    if rc != 0:
        logger.error(f"run_LiFCal_calibration failed (rc={rc}).")
        return

    # 3) locate combined.json (latest run)
    latest = find_latest_run_dir(LIFCAL_OUT_ROOT)
    cpath = combined_json_path(latest)

    if not cpath:
        logger.warning("No combined.json found after LiFCal run.")
        return

    logger.info(f"Latest combined.json: {cpath}")

    # 4) log health + baseline update using your ResultLogger
    ok = try_log_lifcal_with_resultlogger(logger, cpath)
    if not ok:
        logger.info("LiFCal finished, but no baseline/health logging was executed.")


def main():
    """
    @brief Periodic loop that runs LiFCal recalibration on a schedule.
    """
    logger = setup_logger()
    logger.info("Starting periodic LiFCal scheduler.")
    logger.info(f"Interval: {INTERVAL_SECONDS}s")
    logger.info(f"Lock: {LOCK_PATH}")

    while True:
        start = time.time()

        if acquire_lock(logger):
            try:
                logger.info("=== CYCLE START ===")
                run_cycle(logger)
                logger.info("=== CYCLE END ===")
            finally:
                release_lock(logger)

        elapsed = time.time() - start
        sleep_s = max(1.0, INTERVAL_SECONDS - elapsed)
        logger.info(f"Sleeping {sleep_s:.1f}s\n")
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
