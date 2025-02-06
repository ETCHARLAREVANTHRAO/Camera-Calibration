# Camera Calibration Using OpenCV

## Overview
This project performs camera calibration using images of a checkerboard pattern. The calibration process estimates the intrinsic and extrinsic parameters of the camera, which are essential for correcting lens distortion and understanding the camera's perspective.

## Features
- Automatically detects checkerboard corners.
- Computes the camera matrix (intrinsic parameters) and distortion coefficients.
- Computes rotation and translation vectors (extrinsic parameters) for each image.
- Supports both OpenCV-based and manual calibration methods.
- Displays detected corners for visualization (optional).

## Requirements
Make sure you have the following dependencies installed:
```sh
pip install opencv-python numpy
```

## Usage
### 1. Prepare Calibration Images
- Place multiple images of a checkerboard pattern in a folder named `images/`.
- Ensure the checkerboard is visible in different positions and orientations.
- The recommended checkerboard size is **7x6** (inner corners).

### 2. Run the Calibration Script
Execute the following command:
```sh
python camera_calibration.py
```

### 3. Adjust Parameters
Modify these parameters in `camera_calibration.py` as needed:
```python
folder = "images"      # Folder containing calibration images
size = (7, 6)          # Number of inner corners in checkerboard (columns, rows)
square = 10            # Size of each square in mm
opencv = True          # Use OpenCV's calibration (True) or manual method (False)
```

### 4. Calibration Output
After running the script, the output will include:
#### Intrinsic Parameters (Camera Matrix):
```
[[fx  0  cx]
 [0  fy  cy]
 [0   0   1]]
```
- Where:
  - **fx, fy** = Focal lengths (in pixels)
  - **cx, cy** = Principal point (optical center)

#### Distortion Coefficients:
```
[k1, k2, p1, p2, k3]
```
- These correct lens distortions.

#### Extrinsic Parameters (Rotation & Translation):
##### Rotation Matrix for Image 1:
```
[[r11 r12 r13]
 [r21 r22 r23]
 [r31 r32 r33]]
```
##### Translation Vector for Image 1:
```
[tx ty tz]
```

## How It Works
1. Detects checkerboard corners in images.
2. Refines corner positions for better accuracy.
3. Stores 3D world coordinates (object points) and 2D image coordinates (image points).
4. Uses OpenCV's `calibrateCamera()` function (or a manual method) to compute parameters.
5. Prints and saves the calibration results (intrinsic and extrinsic parameters).

## Notes
- The more calibration images you use, the more accurate the results.
- Ensure good lighting and different viewpoints for the best calibration.
- If corner detection fails, try adjusting the size parameter to match your checkerboard.

## License
This project is open-source and can be modified. Suggestions are welcome.

## Author
**Etcharla Revanth Rao**
