# Bikefitting

AI-powered bike fitting analysis. Detects bike angle and rider joint angles from video.

## What's in each folder

**models/** - All the AI models live here. You need these to run anything.

**create_labeled_dataset/** - Tool to create training data by syncing video with phone IMU data.

**bike_angle_detection_model/** - Train and run the bike angle prediction model.

**joint_angle_detection/** - Live webcam joint angle detection.

**bikefitting-web/** - Web app for video analysis (upload a video, get results).

**sam3/** - External library (already installed, don't touch).

## Setup

You need Python 3.10+ and the models folder with all 3 model files.

### Windows

```
python -m venv bikeEnv
bikeEnv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e sam3
```

### Mac/Linux

```
conda create -n bikefitting python=3.12
conda activate bikefitting
./install.sh
```

## Quick start

1. Make sure the models/ folder has all 3 files (best_model.pt, yolov8m-pose.pt, yolov8n-seg.pt)

2. Try live joint detection:
   ```
   cd joint_angle_detection
   python 1_live_camera.py
   ```

3. Or run the web app - see bikefitting-web/README.md
