
# Data-driven Camera Calibration for Light Field Camera Systems

This repository contains the full implementation developed as part of a Bachelor's thesis. It provides a **complete calibration pipeline** for a **cross-shaped light field camera array**, including:

* Target-based **initial calibration**
* Continuous **health monitoring**
* Automatic **target-free recalibration using LiFCal**
* Baseline management and rollback
* Full logging and reproducibility

The system is designed for **Linux** and assumes a **static or quasi-static scene** during calibration.

---

## System Overview

The calibration workflow consists of three phases:

1. **Initial Calibration (Lab / Checkerboard)**
2. **Periodic Health Monitoring**
3. **Target-Free Online Recalibration (LiFCal)**

A single scalar **Health Score (0-100)** represents the quality of the current calibration and is used to decide whether recalibration results are accepted or discarded.

---

## Repository Structure

```text
calibrationLFC/
|--- Controller.py                  # Central orchestration logic
|--- InitialCalibration.py          # OpenCV-based checkerboard calibration
|--- CalibrationState.py            # Calibration state container
|--- ImageSet.py                    # Image set abstraction
|--- HealthMonitor.py               # Health score computation (initial + LiFCal)
|--- ResultLogger.py                # Logging, JSON export, baseline handling
|--- run_initial_calibration.py     # Run initial checkerboard calibration
|--- run_periodic_lifcal.py         # Periodic LiFCal scheduler (daemon-like)
|--- results/                       # Stored calibration results (JSON)
|--- baseline/                      # Baseline calibration (baseline.json + log)
|--- plots/                         # Evaluation scripts (optional)
|--- README.md
```

External LiFCal-related scripts and data are located in:

```text
/data/external/LiFCal/LiFCal_Data/
|--- run_LiFCal_calibration.py  # run LiFCal calibration based on the imageset
|--- run_imageset_creation.py   # creates imageset for the LiFCal calibration (Focus + depthmaps per image)
|--- Recalibration/           
|---|--- LiFCal_Imageset/       
|---|---|---focus/              # Images captured with the light field camera (in this case simulated using Blender)
|---|---|---depth/              # Depthmaps per image made with focused images
|---|---CalibrationByCamera/    # recalibration results per camera and combined
```

---

## Camera Rig Assumptions

* Cross-shaped camera array:

```text
Center
Up1   Up2   Up3
Down1 Down2 Down3
Left1 Left2 Left3
Right1 Right2 Right3
```

* Fixed physical layout
* Synchronized captures
* No moving calibration targets during LiFCal runs

---

## Dependencies (Linux)

### Python Version

```bash
Python = 3.8
```

### Python Packages

```bash
pip install numpy opencv-python scipy
```

| Package       | Purpose                              |
| ------------- | ------------------------------------ |
| numpy         | Numerical computations               |
| opencv-python | Checkerboard detection & calibration |
| scipy         | Bundle adjustment (least squares)    |

### System Dependencies

```bash
sudo apt install -y build-essential cmake libopencv-dev
```

### External Dependency: LiFCal

* Compiled LiFCal binary
* Expected path:

```bash
/data/external/LiFCal/build/bin/LiFCal
```

---

## Phase 1 - Initial Calibration (Checkerboard)

### Image Set Structure

Images must be stored as:

```text
sets/
|--- imagesetX/
    |--- pose_000_Center.png
    |--- pose_000_Down1.png
    |--- pose_000_Down2.png
    |--- ...
```

### Run Initial Calibration

```bash
cd /data/calibrationLFC
python3 run_initial_calibration.py
```

### What Happens

* Checkerboard corners are detected
* Intrinsics and extrinsics are estimated
* Optional bundle adjustment is applied
* Calibration is stored as JSON
* A global Health Score is computed
* If no baseline exists ? `baseline.json` is created automatically
* Calibration is saved in /baseline as calibration_initial_imagesetX_20260116_164546.json

### Log Output

```text
[data/baseline/calibration.log]
2026-01-15 15:52:20,750 [INFO] Stored initial calibration to 'calibration_initial_imageset5_20260115_155220.json' (intrinsics for 13 cams, extrinsics for 13 cams, bundleAdjustf: True).
HealthScore = 92.99 (method=initial)
Baseline created: baseline.json
```

---

## Phase 2 - Health Monitoring

Health monitoring evaluates calibration stability **without using targets**.

### Initial Calibration Health Score

Computed from:

* Mean reprojection error across all cameras
* Mean stereo RMS
* Normalized and mapped to **[0-100]**

This score serves as the **ground-truth baseline**.

---

## Phase 3 - Target-Free Recalibration (LiFCal)

LiFCal-based recalibration follows the same raw image input concept as the initial checkerboard calibration.
Each pose consists of one image per camera, stored in a single shared directory.

Unlike Phase 1, no calibration target is required. Focus images and depth maps are derived internally from this raw input.

### Image Set Creation (Manual)

Images must be stored as:

```text
/data/external/LiFCal/LiFCal_Data/Recalibration/
|--- LiFCal_Imageset/
    |--- pose_000_Center.png
    |--- pose_000_Down1.png
    |--- pose_000_Down2.png
    |--- pose_000_Up1.png
    |--- ...
    |--- pose_001_Center.png
    |--- pose_001_Down1.png
    |--- ...
```

Requirements:

All cameras must be present for each pose

Pose naming (pose_XXX_<Camera>.png) must be consistent

This structure is identical to Phase 1.

### Derived LiFCal Input Data

From the raw image set, LiFCal requires two derived inputs per camera:

- Focused images (sub-aperture / refocused views)

- Depth maps corresponding to each focused image

### Manual Image Set Creation

If no valid LiFCal image set exists, it can be generated manually:

```bash
cd /data/external/LiFCal/LiFCal_Data
python3 run_imageset_creation.py
```

Alternatively, in the periodic calibration system, this is handled automatically.

### Run LiFCal-Calibration Once

```bash
cd /data/external/LiFCal/LiFCal_Data
python3 run_LiFCal_calibration.py
```

### What Happens

- LiFCal is executed independently for each camera
- A directory is created at  
  `/data/external/LiFCal/LiFCal_Data/Recalibration/CalibrationByCamera/`
- Inside this directory:
  - One subfolder is created for each camera
  - Each camera folder contains:
    - `Settings.yaml`
    - `parameters.json` with the estimated camera parameters
- A final `combined.json` file is generated, containing the calibration parameters of all cameras

---

## Periodic Recalibration (Automatic)

### Start Periodic Scheduler

```bash
cd /data/calibrationLFC
python3 run_periodic_lifcal.py
```

### Features

* Runs every *N* seconds (e.g. 300s)
* File lock prevents concurrent runs
* Automatically:

  * Checks image readiness
  * Runs LiFCal
  * Computes health score
  * Updates baseline if improved
  * Logs everything

### Example Log

```text
HealthScore = 58.93 (method=lifcal)
Baseline kept: baseline.json (old=92.99 >= new=58.93)
```

---

## Baseline Management

Stored in:

```bash
/data/baseline/baseline.json
```

### Rules

* Initial calibration always creates the first baseline
* LiFCal updates baseline **only if health score improves**
* Automatic rollback if recalibration is worse

---

## Logging & Reproducibility

All events are logged to:

```bash
/data/baseline/calibration.log
```

Logged information includes:

* Timestamps
* Run type (`initial` / `lifcal`)
* Image sets used
* Health scores
* Baseline decisions

All calibration results are stored as JSON for later evaluation.

---

## Notes

* Plot scripts exist for evaluation but are not required for operation
* The system is designed for research and prototyping
* Real sensor input can later replace synthetic or static datasets

---

## License

This project was developed for **academic research purposes**.
Commercial use requires prior permission.
