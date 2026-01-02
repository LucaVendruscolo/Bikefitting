# BikeFit Pro - Web Application

AI-powered bike fitting analysis that runs **entirely in your browser**.

## Features

- 📹 Upload cycling videos for analysis
- ✂️ Select start/end times and output frame rate
- 📊 Real-time processing with progress tracking
- 🎬 Video playback with timeline controls
- 📐 Joint angle detection (knee, hip, elbow)
- 🚲 Bike angle prediction
- ⏱️ Processing metrics
- 🔒 **100% client-side** - your video never leaves your device

## How It Works

All ML models run directly in your browser using ONNX Runtime Web:
- Pose estimation: YOLOv8-pose (ONNX)
- Bike segmentation: YOLOv8-seg (ONNX)
- Bike angle: ConvNeXT classifier (ONNX)

## Setup

### 1. Convert Models to ONNX

Before deploying, you need to convert the trained PyTorch models to ONNX format:

```bash
cd web
pip install torch torchvision ultralytics onnx
python convert_models.py
```

This creates ONNX models in `public/models/`:
- `bike_angle.onnx` - Bike angle classifier
- `bike_angle_config.json` - Model configuration
- `yolov8-pose.onnx` - Pose estimation
- `yolov8-seg.onnx` - Bike segmentation

### 2. Install Dependencies

```bash
npm install
```

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### 4. Build for Production

```bash
npm run build
```

## Deployment (Vercel)

1. Push code to GitHub
2. Connect repo to Vercel
3. Set root directory to `web`
4. Deploy!

**Note:** Make sure to run `convert_models.py` and commit the ONNX models to `public/models/` before deploying.

## Model Requirements

The ONNX models in `public/models/` are required:
- `bike_angle.onnx` (~30MB)
- `bike_angle_config.json` (~1KB)
- `yolov8-pose.onnx` (~25MB)
- `yolov8-seg.onnx` (~25MB)

Total size: ~80MB (loaded on first use)

## Browser Support

Works best in:
- Chrome/Edge (WebGL acceleration)
- Firefox
- Safari (limited WebGL support)

## Future Development

The sidebar includes a placeholder for **seat height and handlebar recommendations**.
These calculations will be added in a future update.

## Tech Stack

- **Frontend**: Next.js 14, React 18, Framer Motion
- **ML Inference**: ONNX Runtime Web
- **Models**: ConvNeXT (bike angle), YOLOv8 (pose & segmentation)
