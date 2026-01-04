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

## Requirements

The model file yolov8m-pose.pt should be in the models/ folder at the project root. If it's missing, the code will auto-download it.
