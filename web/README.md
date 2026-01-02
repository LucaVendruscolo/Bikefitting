# BikeFitting Web App

A client-side web application for AI-powered bike fit analysis. All processing happens in the browser - your video never leaves your device.

## Features

- Upload cycling videos for analysis
- AI-powered body pose detection (YOLOv8m-pose)
- Bike angle estimation (ConvNeXt)
- Joint angle measurements (knee, hip, elbow)
- Works on phone and desktop browsers
- No server required - runs 100% client-side

## Setup

### 1. Install dependencies

```bash
cd web
npm install
```

### 2. Convert models to ONNX

Run the conversion script (requires Python with PyTorch):

```bash
python convert_models_to_onnx.py
```

This creates ONNX models in `public/models/`:
- `yolov8m-pose.onnx` - Body pose detection (~52MB)
- `yolov8n-seg.onnx` - Bike segmentation (~7MB)  
- `bike_angle.onnx` - Bike angle estimation (~30MB)

### 3. Run locally

```bash
npm run dev
```

Open http://localhost:3000

### 4. Deploy to Vercel

```bash
# Push to GitHub, then:
vercel
```

Or connect your GitHub repo to Vercel for automatic deployments.

## Project Structure

```
web/
├── public/
│   └── models/           # ONNX models (generated)
├── src/
│   ├── app/
│   │   ├── page.tsx      # Main page
│   │   ├── layout.tsx    # Root layout
│   │   └── globals.css   # Global styles
│   ├── components/
│   │   ├── VideoUploader.tsx
│   │   ├── VideoProcessor.tsx
│   │   └── ResultsPanel.tsx
│   └── lib/
│       └── inference.ts  # ONNX Runtime inference
├── convert_models_to_onnx.py
├── package.json
└── README.md
```

## Browser Compatibility

Tested on:
- Chrome 90+ (recommended)
- Firefox 90+
- Safari 15+
- Edge 90+

WebGL or WebGPU support required for best performance.

## Future Plans

- [ ] Bike fit recommendations based on joint angles
- [ ] Camera angle correction using bike orientation
- [ ] Export analysis report
- [ ] Comparison with ideal angles

