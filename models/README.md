# Models

This folder contains all the AI models needed to run the bike fitting analysis.

## Required files

**best_model.pt** (284 MB)
Your trained bike angle prediction model. Predicts how the bike is angled relative to the camera (-180 to 180 degrees).

**yolov8m-pose.pt** (51 MB)
Human pose detection model. Finds the rider's body joints (shoulders, elbows, hips, knees, ankles).

**yolov8n-seg.pt** (7 MB)
Bike segmentation model. Masks out everything except the bike in each frame.

## If you're missing these files

The YOLO models (yolov8m-pose.pt and yolov8n-seg.pt) will auto-download when you first run the code if they're missing.

The best_model.pt file is your custom trained model. If you don't have it, you need to either:
- Get it from whoever trained it
- Train your own using bike_angle_detection_model/

## For the web app

If you're running the web app on Modal, you also need to upload best_model.pt to Modal's volume:

```
modal volume create bikefitting-models
modal volume put bikefitting-models best_model.pt best_model.pt
```

