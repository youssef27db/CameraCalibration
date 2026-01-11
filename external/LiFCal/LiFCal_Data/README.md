# Introduction

This data is associated with the LiFCal repository (https://github.com/RobotVisionHKA/LiFCal.git) for light field camera calibration. It contains some example data for a first program run.

# Data structure

This repository contains the following folders:
- Calibration_Marker: Contains data for full camera calibration, using markers only for scaling.
- Recalibration: Contains data for recalibration of the camera by fixing the parameters f (focal length) and B (distance between MLA and sensor).

In the main folders, each sequence is grouped in a subfolder. The data provided is as follows:
- Images\_Raw\_Processed: folder containing raw camera images after debayering.
- Images\_Focused: folder with totally focused images.
- Images\_Depth\_Filled: folder with depth maps of images after partial filling.
- MLA\_Calibration.xml: MLA calibration description file.
- Settings.yaml: calibration settings file, to be entered as an argument to the executable file.
- Constraints.txt: distance constraints between markers to set the scene scale, to be entered as an argument to the executable file.
- Fixed\_Paramerters.txt: file of parameters set for recalibration, to be entered as an argument to the executable file.

The "Settings.yaml" file contains paths to the program executable file. To avoid having to modify the settings file, the complete repository can be cloned at the root of the LiFCal folder. Otherwise, the paths for the parameters `Path.totalFocusImages`, `Path.microLensCalibration` and `Path.virtualDepthData` parameters will have to be adapted accordingly.

# Additional data

This repository also contains a “Markers\_DICT\_6X6\_250.pdf” file. These are the Aruco markers used for the calibration with markers.

This document was produced using markers from the OpenCV dictionary with index 10. The dictionary contains 250 different markers with a size of 6 squares (not including the black outline). The document contains the index of each marker printed in the dictionary (from 0 to 4).

The document also contains arrows to indicate the spacing between markers. Distances are given from marker center to marker center. To ensure that these distances are correct when printed, the print scale must be set to 100% (and not left at the default value, which adds an edge and distorts the scale). Otherwise, the distances in the constraints file will have to be adapted to actual dimensions.

