# BikeFit AI

A web app that analyzes cycling videos to detect body posture and bike angles using AI.

Upload a video → AI analyzes your position → Get real-time angle measurements

## How It Works

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Your Browser  │────▶│  Next.js App    │────▶│  Modal (GPU)    │
│                 │     │  (Vercel)       │     │                 │
│  Upload Video   │     │  API Proxy      │     │  AI Processing  │
│  View Results   │◀────│  Serve Frontend │◀────│  YOLO + ConvNeXt│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

1. **Upload**: Select a cycling video and choose time range/FPS
2. **Process**: Video is sent to Modal's GPU servers for AI analysis
3. **Stream**: Real-time progress updates as frames are processed
4. **View**: Watch the annotated video with live angle measurements

## AI Models

| Model | Purpose |
|-------|---------|
| **YOLOv8n-seg** | Segments the bike from the frame |
| **YOLOv8m-pose** | Detects human body keypoints |
| **ConvNeXt** | Predicts bike tilt angle (your trained model) |

## Measurements

- **Bike Angle**: Tilt of the bike (from your custom model)
- **Knee Angle**: Angle at the knee joint
- **Hip Angle**: Angle at the hip joint  
- **Elbow Angle**: Angle at the elbow joint

## Project Structure

```
bikefitting-web/
├── backend/                    # Modal backend (GPU processing)
│   ├── modal_app.py           # Main API endpoints
│   └── processing/            # Local copies of processing modules
│
├── frontend/                   # Next.js frontend (Vercel)
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx       # Main page
│   │   │   └── api/           # API routes
│   │   │       └── process-stream/  # Proxy to Modal
│   │   └── components/
│   │       ├── VideoUploader.tsx    # Upload + settings UI
│   │       └── ResultsViewer.tsx    # Video player + live data
│   └── package.json
│
└── SETUP.md                    # Detailed deployment guide
```

## Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) 18+
- [Modal](https://modal.com/) account
- Your trained model file (`best_model.pt`)

### 1. Deploy Backend (Modal)

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal setup

# Create model storage and upload your model
modal volume create bikefitting-models
modal volume put bikefitting-models path/to/best_model.pt best_model.pt

# Create API key secret
modal secret create bikefitting-api-key BIKEFITTING_API_KEY=your-secret-key-here

# Deploy
cd backend
modal deploy modal_app.py
```

### 2. Run Frontend

```bash
cd frontend
npm install

# Create .env.local with your Modal URLs (replace YOUR-USERNAME)
echo "MODAL_STREAM_URL=https://YOUR-USERNAME--bikefitting-process-video-stream.modal.run" > .env.local
echo "MODAL_API_KEY=your-secret-key-here" >> .env.local
echo "NEXT_PUBLIC_MODAL_DOWNLOAD_URL=https://YOUR-USERNAME--bikefitting-download.modal.run" >> .env.local

npm run dev
```

Open http://localhost:3000

### 3. Deploy to Vercel

```bash
npm install -g vercel
vercel

# Set environment variables in Vercel dashboard:
# - MODAL_STREAM_URL
# - MODAL_API_KEY
# - NEXT_PUBLIC_MODAL_DOWNLOAD_URL
```

## Environment Variables

### Frontend (.env.local)

| Variable | Description |
|----------|-------------|
| `MODAL_STREAM_URL` | URL to Modal's streaming endpoint (server-side) |
| `MODAL_API_KEY` | Secret key for API authentication (server-side) |
| `NEXT_PUBLIC_MODAL_DOWNLOAD_URL` | URL to Modal's download endpoint (client-side) |

### Backend (Modal Secrets)

| Secret Name | Key | Description |
|-------------|-----|-------------|
| `bikefitting-api-key` | `BIKEFITTING_API_KEY` | Must match frontend's `MODAL_API_KEY` |

## API Endpoints

### Modal Backend

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/process_video_stream` | POST | Process video with SSE progress updates |
| `/download/{job_id}` | GET | Download processed video |
| `/health` | GET | Health check |

### Request Format

```json
{
  "video_base64": "...",
  "api_key": "your-key",
  "output_fps": 10,
  "start_time": 0,
  "end_time": 30,
  "max_duration_sec": 120
}
```

### SSE Response Events

```
data: {"type": "progress", "stage": "processing", "message": "Analyzing...", "percent": 50}
data: {"type": "complete", "job_id": "abc123", "stats": {...}, "frame_data": [...]}
```

## Limits

| Limit | Value |
|-------|-------|
| Max video duration | 2 minutes |
| Max file size | 200 MB |
| Output FPS range | 5-15 |
| Rate limit | 10 requests/hour |

## Troubleshooting

### Video not loading
- Check browser console for errors
- Ensure Modal endpoint is deployed and running
- Verify API key matches in frontend and backend

### Slow processing
- Reduce output FPS (5 is fastest)
- Select a shorter time range
- Modal cold starts take ~30s on first request

## Development

```bash
# Backend - test locally
cd backend
modal serve modal_app.py  # Runs with hot-reload

# Frontend - development server
cd frontend
npm run dev
```
