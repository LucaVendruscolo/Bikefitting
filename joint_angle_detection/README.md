Joint angle detection

What this does

- Uses YOLOv8 pose to detect a rider’s joints from a webcam feed
- Computes 3 angles on one body side:
  - knee (hip-knee-ankle)
  - hip (shoulder-hip-knee)
  - elbow (shoulder-elbow-wrist)
- Draws the joints and angles on the live video

Folder setup (needed before running)

- `joint_angle_detection/models/yolov8n-pose.pt` must exist
  - This repo includes it already. If it’s missing, download the YOLOv8 pose weights and put it there.

Run

```bash
conda activate bikefitting
cd joint_angle_detection
python 1_live_camera.py
```

Options

- Choose camera: `--source 0`
- Force side: `--side left` or `--side right` (default is auto)
- Keypoint confidence: `--min_conf 0.5`
- Custom weights path: `--model path\\to\\yolov8n-pose.pt`

Notes

- Press `q` to quit.

