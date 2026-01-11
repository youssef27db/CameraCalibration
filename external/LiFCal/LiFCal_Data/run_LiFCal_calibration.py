import os
import re
import shutil
import subprocess
from datetime import datetime

# CONFIG
LIFCAL_BIN = "/data/external/LiFCal/build/bin/LiFCal"

SETTINGS = "/data/external/LiFCal/LiFCal_Data/Recalibration/Settings.yaml"
FIXED_PARAMS      = "/data/external/LiFCal/LiFCal_Data/Recalibration/Fixed_Paramerters.txt"

# MLA-Pfad
MLA_CALIB_XML     = "/data/external/LiFCal/LiFCal_Data/Recalibration/MLA_Calibration.xml"

ROOT_IMAGESET = "/data/external/LiFCal/LiFCal_Data/Recalibration/LiFCal_Imageset"
FOCUS_ROOT = os.path.join(ROOT_IMAGESET, "focus")   # focus/<CamId>/
DEPTH_ROOT = os.path.join(ROOT_IMAGESET, "depth")   # depth/<CamId>/

OUT_ROOT = "/data/external/LiFCal/LiFCal_Data/Recalibration/CalibrationByCamera"

CAMERA_IDS = [
    "Center", "Up1", "Up2", "Up3",
    "Down1", "Down2", "Down3",
    "Left1", "Left2", "Left3",
    "Right1", "Right2", "Right3"
]

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()

def write_text(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

def _replace_yaml_key_line(text: str, key: str, new_val: str) -> str:
    """
    Ersetzt Zeilen wie:
      Key: something
      Key: "something"
    durch:
      Key: "new_val"
    """
    pattern = re.compile(rf"^({re.escape(key)}\s*:\s*)(.*)$", re.MULTILINE)
    return pattern.sub(rf'\1"{new_val}"', text)

def patch_settings_yaml(template_text: str, focus_dir: str, depth_dir: str, mla_xml: str) -> str:
    out = template_text
    out = _replace_yaml_key_line(out, "Path.totalFocusImages", focus_dir)
    out = _replace_yaml_key_line(out, "Path.virtualDepthData", depth_dir)
    out = _replace_yaml_key_line(out, "Path.microLensCalibration", mla_xml)
    return out

def run_lifcal_recalib(settings_path: str, fixed_params_path: str, store_dir: str) -> int:
    os.makedirs(store_dir, exist_ok=True)

    cmd = [LIFCAL_BIN, "recalib", settings_path, fixed_params_path]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # alles in stdout loggen (hilft debugging)
        text=True,
        cwd=os.path.dirname(LIFCAL_BIN),  
    )

    out, _ = proc.communicate(input=f"y\n{store_dir}\n")
    write_text(os.path.join(store_dir, "lifcal_stdout.log"), out)

    return proc.returncode

def find_results_folder(store_dir: str) -> str:
    """
    LiFCal legt meist einen Unterordner Calibration_Results_... an.
    Wenn nicht: store_dir selbst.
    """
    if not os.path.isdir(store_dir):
        raise RuntimeError(f"store_dir existiert nicht: {store_dir}")

    subdirs = []
    for name in os.listdir(store_dir):
        p = os.path.join(store_dir, name)
        if os.path.isdir(p):
            subdirs.append(p)

    if not subdirs:
        return store_dir

    subdirs.sort(key=lambda p: os.path.getmtime(p))
    return subdirs[-1]


def copy_essentials(result_dir: str, out_cam_dir: str) -> None:
    proto = os.path.join(result_dir, "calibrationProtocol.txt")
    extr  = os.path.join(result_dir, "extrinsicOrientations.xml")

    missing = []
    if not os.path.isfile(proto):
        missing.append("calibrationProtocol.txt")
    if not os.path.isfile(extr):
        missing.append("extrinsicOrientations.xml")
    if missing:
        raise RuntimeError(f"Fehlende Dateien in {result_dir}: {missing}")

    shutil.copy2(proto, os.path.join(out_cam_dir, "calibrationProtocol.txt"))
    shutil.copy2(extr,  os.path.join(out_cam_dir, "extrinsicOrientations.xml"))

def main():
    if not os.path.isfile(LIFCAL_BIN):
        raise SystemExit(f"LiFCal binary nicht gefunden: {LIFCAL_BIN}")
    if not os.path.isfile(SETTINGS):
        raise SystemExit(f"Settings template nicht gefunden: {SETTINGS}")
    if not os.path.isfile(FIXED_PARAMS):
        raise SystemExit(f"Fixed parameters file nicht gefunden: {FIXED_PARAMS}")
    if not os.path.isfile(MLA_CALIB_XML):
        raise SystemExit(f"MLA_Calibration.xml nicht gefunden: {MLA_CALIB_XML}")

    templ = read_text(SETTINGS)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    session_out = os.path.join(OUT_ROOT, f"run_{timestamp}")
    os.makedirs(session_out, exist_ok=True)

    print("Output:", session_out)

    for cam in CAMERA_IDS:
        focus_dir = os.path.join(FOCUS_ROOT, cam)
        depth_dir = os.path.join(DEPTH_ROOT, cam)

        if not os.path.isdir(focus_dir):
            print(f"[SKIP] {cam}: focus dir fehlt: {focus_dir}")
            continue
        if not os.path.isdir(depth_dir):
            print(f"[SKIP] {cam}: depth dir fehlt: {depth_dir}")
            continue

        cam_out = os.path.join(session_out, cam)
        os.makedirs(cam_out, exist_ok=True)

        settings_cam = os.path.join(cam_out, "Settings.yaml")
        patched = patch_settings_yaml(templ, focus_dir, depth_dir, MLA_CALIB_XML)
        write_text(settings_cam, patched)

        store_dir = os.path.join(cam_out, "lifcal_store")
        rc = run_lifcal_recalib(settings_cam, FIXED_PARAMS, store_dir)

        if rc != 0:
            print(f"[FAIL] {cam}: LiFCal returncode={rc} (siehe {store_dir}/lifcal_stdout.log)")
            continue

        result_dir = find_results_folder(store_dir)

        try:
            copy_essentials(result_dir, cam_out)
        except Exception as e:
            print(f"[FAIL] {cam}: {e}")
            print(f"        check log: {store_dir}/lifcal_stdout.log")
            continue

        # store dir wegräumen
        shutil.rmtree(store_dir, ignore_errors=True)

        print(f"[OK] {cam}: gespeichert in {cam_out}")

    print("DONE.")

if __name__ == "__main__":
    main()
