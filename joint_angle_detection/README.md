# Joint Angle Detection

Live webcam detection of rider body angles using pose estimation.

## What it does

Points a camera at someone and it detects their joints in real-time. It calculates three angles:
- Knee angle (hip to knee to ankle)
- Hip angle (shoulder to hip to knee)
- Elbow angle (shoulder to elbow to wrist)

## How to run

```
cd joint_angle_detection
python 1_live_camera.py
```

Press 'q' to quit.

## Options

Use a different camera:
```
python 1_live_camera.py --source 1
```

Force left or right side (default auto-detects):
```
python 1_live_camera.py --side left
```

Adjust detection sensitivity:
```
python 1_live_camera.py --min_conf 0.3
```
There are scripts for running this from a locally saved video
```
python 1b_video_analytic.py --input path/to/video
```
Or from a csv including a list of frames such as that created by bike_angle_detection_model and create_labeled_dataset
```
python 1b_csv_analytic.py 
```

Further we have 
```
python 2_fill_crank_angle.py --min_conf .8
```
Which applies the Gaussian Process to low-confidence foot measurements

```
python o_plot_angles.py
```
Which provides some charts of the relevant gaussian processes. 

## Requirements

The model file yolov8m-pose.pt should be in the models/ folder at the project root. If it's missing, the code will auto-download it.
