import json
import xml.etree.ElementTree as ET
import re
from pathlib import Path


BASE_DIR = Path(
    "/data/external/LiFCal/LiFCal_Data/Recalibration/"
    "center_calibration/Calibration_Results_2026_01_09_142929"
)

CAMERA_ID = "Center"
OUTPUT_JSON = "/data/external/LiFCal/results/center_calibration.json"

PROTOCOL_FILE = BASE_DIR / "calibrationProtocol.txt"
EXTRINSIC_XML = BASE_DIR / "extrinsicOrientations.xml"

# calibrationProtocol.txt parsen
def parse_protocol(path):
    txt = path.read_text()

    def grab(name):
        m = re.search(rf"{name}\s*:\s*([-+eE0-9\.]+)", txt)
        return float(m.group(1)) if m else None

    intrinsics = {
        "pixelSize": grab("Pixel Size"),
        "fL": grab("fL"),
        "bL0": grab("bL0"),
        "B": grab("B"),
        "cx": grab("cx"),
        "cy": grab("cy"),
        "a0": grab("a0"),
        "a1": grab("a1"),
        "b0": grab("b0"),
        "b1": grab("b1"),
    }

    reprojection = {
        "stdDevX": grab("std. Dev. x"),
        "stdDevY": grab("std. Dev. y"),
        "maeX": grab("mae x"),
        "maeY": grab("mae y"),
    }

    return intrinsics, reprojection

# extrinsicOrientations.xml parsen
def parse_extrinsics(path):
    tree = ET.parse(path)
    root = tree.getroot()

    poses = {}

    for frame in root.findall("Frame"):
        frame_id = frame.attrib["id"]

        rot = [float(c.text) for c in frame.find("Rotation").findall("Coeff")]
        trans = [float(c.text) for c in frame.find("Translation").findall("Coeff")]

        poses[f"Frame_{frame_id}"] = {
            "rotationRodrigues": rot,     # exakt wie LiFCal
            "translation": trans
        }

    return poses

# JSON bauen
intrinsics, reproj = parse_protocol(PROTOCOL_FILE)
poses = parse_extrinsics(EXTRINSIC_XML)

result = {
    "meta": {
        "runType": "recalib_single_camera",
        "cameraId": CAMERA_ID,
        "numPoses": len(poses),
    },
    "state": {
        "intrinsics": {
            CAMERA_ID: {
                "model": "LiFCal",
                "parameters": intrinsics,
                "reprojectionError": reproj
            }
        },
        "poses": poses,
        "timeStamp": None
    }
}

# schreiben
Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(result, f, indent=2)

print(f"[OK] JSON geschrieben nach: {OUTPUT_JSON}")
print(f"[OK] Posen: {len(poses)}")
