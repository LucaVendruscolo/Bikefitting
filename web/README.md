# BikeFit Pro - Web Application

AI-powered bike fitting analysis using computer vision.

## Architecture

- **Frontend**: Next.js 14 with React 18, deployed on Vercel
- **Backend**: FastAPI Python server for ML inference

## Features

- 📹 Upload cycling videos for analysis
- ✂️ Select start/end times and output frame rate
- 📊 Real-time processing with progress tracking
- 🎬 YouTube-style video playback with timeline controls
- 📐 Joint angle detection (knee, hip, elbow)
- 🚲 Bike angle prediction
- ⏱️ Processing metrics for optimization
- 🎨 Futuristic, Apple-inspired design

## Quick Start

### 1. Install Frontend Dependencies

```bash
cd web
npm install
```

### 2. Install Backend Dependencies

```bash
cd web/backend
pip install -r requirements.txt
```

Make sure you also have the main project dependencies:
```bash
cd Bikefitting
pip install -r requirements.txt
```

### 3. Start the Backend

```bash
cd web/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the Frontend (in another terminal)

```bash
cd web
npm run dev
```

### 5. Open in Browser

Visit [http://localhost:3000](http://localhost:3000)

## Processing Pipeline

The web app uses the **exact same pipeline** as `generate_demo_video.py`:

1. **Video Upload** → Frames extracted at specified FPS
2. **Bike Segmentation** → YOLOv8-seg masks bicycle pixels
3. **Bike Angle Prediction** → ConvNeXT classifier (120 bins, circular encoding)
4. **Pose Detection** → YOLOv8m-pose for 17 COCO keypoints
5. **Joint Angles** → Computed from detected keypoints (knee, hip, elbow)
6. **Keypoint Smoothing** → Exponential moving average (α=0.6)

This ensures predictions are consistent with training.

## Required Models

The backend expects these pretrained models:

- `Bikefitting/bike_angle_detection_model/models/optuna_best/best_model.pt`
- `Bikefitting/bike_angle_detection_model/yolov8n-seg.pt`
- `Bikefitting/joint_angle_detection/models/yolov8m-pose.pt` (or will auto-download)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Check server status and model loading |
| `/api/process` | POST | Process video with streaming progress |
| `/api/video/{filename}` | GET | Serve processed video files |

## Processing Metrics

Each processed video includes timing metrics:

- **Segmentation Time**: Average time for bike masking per frame
- **Pose Detection Time**: Average time for keypoint detection per frame
- **Bike Angle Time**: Average time for angle classification per frame
- **Total Time**: Complete processing duration

## Future Development

The sidebar includes a placeholder for **seat height and handlebar recommendations**.
These calculations will be added in a future update.

## Deployment

### Frontend (Vercel)

The `web` folder is configured for Vercel deployment:

1. Connect your GitHub repo to Vercel
2. Set root directory to `web`
3. Deploy automatically on push

### Backend

The Python backend requires GPU for optimal performance. Options:

- **Railway** or **Render** for managed deployment
- **AWS EC2** or **GCP Compute** with GPU
- **Modal** or **Replicate** for serverless GPU

Set the `API_URL` environment variable in your Vercel deployment to point to your backend.

## Tech Stack

- **Frontend**: Next.js 14, React 18, Framer Motion, TypeScript
- **Backend**: FastAPI, PyTorch, OpenCV, Ultralytics YOLO
- **ML Models**: ConvNeXT (bike angle), YOLOv8 (segmentation & pose)

