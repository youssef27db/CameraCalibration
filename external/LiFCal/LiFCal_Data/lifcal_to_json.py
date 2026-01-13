import os
import re
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional

import numpy as np
import cv2


def _parse_protocol(protocol_path: str) -> dict:
    txt = open(protocol_path, "r", encoding="utf-8", errors="ignore").read()

    def find_float(pattern: str, default=None) -> Optional[float]:
        m = re.search(pattern, txt, re.MULTILINE)
        if not m:
            return default
        try:
            return float(m.group(1))
        except Exception:
            return default

    pixel_size = find_float(r"Pixel Size:\s*([0-9.+-eE]+)\s*mm", None)

    fL  = find_float(r"^\s*fL\s*:\s*([0-9.+-eE]+)\s*$", None)
    bL0 = find_float(r"^\s*bL0\s*:\s*([0-9.+-eE]+)\s*$", None)
    B   = find_float(r"^\s*B\s*:\s*([0-9.+-eE]+)\s*$", None)
    cx  = find_float(r"^\s*cx\s*:\s*([0-9.+-eE]+)\s*$", None)
    cy  = find_float(r"^\s*cy\s*:\s*([0-9.+-eE]+)\s*$", None)

    a0 = find_float(r"^\s*a0\s*:\s*([0-9.+-eE]+)\s*$", 0.0)
    a1 = find_float(r"^\s*a1\s*:\s*([0-9.+-eE]+)\s*$", 0.0)
    b0 = find_float(r"^\s*b0\s*:\s*([0-9.+-eE]+)\s*$", 0.0)
    b1 = find_float(r"^\s*b1\s*:\s*([0-9.+-eE]+)\s*$", 0.0)

    x_std = find_float(r"std\.\s*Dev\.\s*x:\s*([0-9.+-eE]+)", None)
    y_std = find_float(r"std\.\s*Dev\.\s*y:\s*([0-9.+-eE]+)", None)
    x_mae = find_float(r"mae\s*x:\s*([0-9.+-eE]+)", None)
    y_mae = find_float(r"mae\s*y:\s*([0-9.+-eE]+)", None)

    return {
        "pixelSizeMm": pixel_size,
        "fL_mm": fL,
        "bL0_mm": bL0,
        "B_mm": B,
        "cx_px": cx,
        "cy_px": cy,
        "a0": a0,
        "a1": a1,
        "b0": b0,
        "b1": b1,
        "reprojection": {
            "x_std": x_std,
            "y_std": y_std,
            "x_mae": x_mae,
            "y_mae": y_mae
        }
    }


def _parse_extrinsics_xml(xml_path: str) -> Dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    out: Dict[str, Any] = {}

    for frame in root.findall("Frame"):
        fid = int(frame.attrib.get("id"))
        pose_key = "pose%03d" % fid

        rot = frame.find("Rotation")
        tra = frame.find("Translation")

        rvec = [0.0, 0.0, 0.0]
        tvec = [0.0, 0.0, 0.0]

        if rot is not None:
            for c in rot.findall("Coeff"):
                i = int(c.attrib["i"])
                rvec[i] = float(c.text.strip())

        if tra is not None:
            for c in tra.findall("Coeff"):
                i = int(c.attrib["i"])
                tvec[i] = float(c.text.strip())

        rvec_np = np.array(rvec, dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(rvec_np)

        out[pose_key] = {
            "rotationMatrix": R.astype(float).tolist(),
            "rotationVector": [[float(rvec[0])], [float(rvec[1])], [float(rvec[2])]],
            "translationVector": [[float(tvec[0])], [float(tvec[1])], [float(tvec[2])]],
        }

    return out


def export_lifcal_parameters_json_in_place(
    result_dir: str,
    cam_id: str,
    image_dir: str,
    run_type: str = "recalib",
    pixel_size_mm_fallback: float = 0.0055,
    protocol_name: str = "calibrationProtocol.txt",
    extr_name: str = "extrinsicOrientations.xml",
    out_name: str = "parameters.json",
    # Alias (damit dein Runner nix ändern muss)
    pixel_size_mm: Optional[float] = None,
    fallback_cx: Optional[float] = None,  # wird nur akzeptiert, nicht genutzt
    fallback_cy: Optional[float] = None,  # wird nur akzeptiert, nicht genutzt
) -> str:
    """
    result_dir: Ordner mit calibrationProtocol.txt + extrinsicOrientations.xml
    schreibt: result_dir/parameters.json

    Wichtig: fallback_cx/fallback_cy werden nur akzeptiert (Kompatibilität),
    weil wir keine KameraMatrix mehr speichern.
    """
    if pixel_size_mm is not None:
        pixel_size_mm_fallback = float(pixel_size_mm)

    protocol_path = os.path.join(result_dir, protocol_name)
    extr_path = os.path.join(result_dir, extr_name)

    if not os.path.isfile(protocol_path):
        raise FileNotFoundError(protocol_path)
    if not os.path.isfile(extr_path):
        raise FileNotFoundError(extr_path)

    protocol = _parse_protocol(protocol_path)
    extr_by_pose = _parse_extrinsics_xml(extr_path)

    # Pixel size fallback
    px = protocol.get("pixelSizeMm", None)
    if px is None or (not np.isfinite(px)) or px <= 0:
        protocol["pixelSizeMm"] = float(pixel_size_mm_fallback)

    # kompakter reprojectionError (avg mae)
    rep = protocol.get("reprojection", {})
    x_mae = rep.get("x_mae", None)
    y_mae = rep.get("y_mae", None)
    avg_mae = None
    if x_mae is not None and y_mae is not None and np.isfinite(x_mae) and np.isfinite(y_mae):
        avg_mae = float((x_mae + y_mae) / 2.0)

    data = {
        "meta": {
            "runType": run_type,
            "numPoses": int(len(extr_by_pose)),
            "imageDir": image_dir,
            "cameraIds": [cam_id],
            "bundleAdjust": True,
        },
        "state": {
            "parameters": {
                cam_id: {
                    "pixelSizeMm": protocol.get("pixelSizeMm"),
                    "fL_mm": protocol.get("fL_mm"),
                    "bL0_mm": protocol.get("bL0_mm"),
                    "B_mm": protocol.get("B_mm"),
                    "cx_px": protocol.get("cx_px"),
                    "cy_px": protocol.get("cy_px"),
                    "a0": protocol.get("a0"),
                    "a1": protocol.get("a1"),
                    "b0": protocol.get("b0"),
                    "b1": protocol.get("b1"),
                    "reprojection": rep,
                    "reprojectionAvgMae": avg_mae,
                }
            },
            "extrinsicsByPose": {cam_id: extr_by_pose},
            "timeStamp": None
        }
    }

    out_path = os.path.join(result_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_path


def build_combined_parameters_json(run_root: str, camera_ids: list, out_name: str = "combined.json") -> str:
    """
    Erwartet pro Kamera: run_root/<CamId>/parameters.json
    """
    combined = {
        "meta": {
            "runType": "combined",
            "cameraIds": camera_ids,
        },
        "state": {
            "parameters": {},
            "extrinsicsByPose": {},
            "timeStamp": None
        }
    }

    found_any = False

    for cam in camera_ids:
        cam_dir = os.path.join(run_root, cam)
        jpath = os.path.join(cam_dir, "parameters.json")

        if not os.path.isfile(jpath):
            continue

        with open(jpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        st = data.get("state", {})
        params = st.get("parameters", {})
        extr_pose = st.get("extrinsicsByPose", {})

        if cam in params:
            combined["state"]["parameters"][cam] = params[cam]
            found_any = True

        if cam in extr_pose:
            combined["state"]["extrinsicsByPose"][cam] = extr_pose[cam]
            found_any = True

    if not found_any:
        raise RuntimeError("No parameters.json found to combine.")

    out_path = os.path.join(run_root, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    return out_path
