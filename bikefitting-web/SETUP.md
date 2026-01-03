# BikeFit AI - Deployment Guide

Complete setup instructions for deploying the BikeFitting web application.

## Prerequisites

- Python 3.9+
- Node.js 18+
- [Modal](https://modal.com/) account (free tier available)
- [Vercel](https://vercel.com/) account (optional, for production)
- Your trained model file (`best_model.pt`)

---

## Step 1: Deploy Backend (Modal)

### 1.1 Install Modal CLI

```bash
pip install modal
modal setup  # Follow prompts to authenticate
```

### 1.2 Upload Your Model

```bash
# Create a volume for persistent model storage
modal volume create bikefitting-models

# Upload your trained model
modal volume put bikefitting-models /path/to/your/best_model.pt best_model.pt
```

### 1.3 Create API Key Secret

Generate a secure API key and store it in Modal:

**PowerShell (Windows):**
```powershell
# Generate a random key
$apiKey = [Convert]::ToHexString([System.Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
Write-Host "Your API key: $apiKey"

# Create the secret in Modal
modal secret create bikefitting-api-key BIKEFITTING_API_KEY=$apiKey
```

**Bash (Mac/Linux):**
```bash
# Generate and create secret
API_KEY=$(openssl rand -hex 32)
echo "Your API key: $API_KEY"
modal secret create bikefitting-api-key BIKEFITTING_API_KEY=$API_KEY
```

**Save this API key** - you'll need it for the frontend.

### 1.4 Deploy to Modal

```bash
cd bikefitting-web/backend

# Windows: Set UTF-8 encoding to avoid display issues
$env:PYTHONUTF8="1"; modal deploy modal_app.py

# Mac/Linux
modal deploy modal_app.py
```

After deployment, note your endpoint URLs (shown in output):
- `https://YOUR-USERNAME--bikefitting-process-video-stream.modal.run`
- `https://YOUR-USERNAME--bikefitting-download.modal.run`

---

## Step 2: Set Up Frontend

### 2.1 Install Dependencies

```bash
cd bikefitting-web/frontend
npm install
```

### 2.2 Configure Environment

Create `.env.local`:

```env
MODAL_STREAM_URL=https://YOUR-USERNAME--bikefitting-process-video-stream.modal.run
MODAL_API_KEY=your-api-key-from-step-1.3
NEXT_PUBLIC_MODAL_DOWNLOAD_URL=https://YOUR-USERNAME--bikefitting-download.modal.run
```

### 2.3 Run Locally

```bash
npm run dev
```

Open http://localhost:3000

---

## Step 3: Deploy to Production (Vercel)

### 3.1 Deploy

```bash
npm install -g vercel
cd bikefitting-web/frontend
vercel
```

### 3.2 Set Environment Variables

In Vercel Dashboard → Project Settings → Environment Variables:

| Name | Value |
|------|-------|
| `MODAL_STREAM_URL` | Your Modal streaming endpoint |
| `MODAL_API_KEY` | Your API key |
| `NEXT_PUBLIC_MODAL_DOWNLOAD_URL` | Your Modal download endpoint |

### 3.3 Redeploy

```bash
vercel --prod
```

---

## Your Model Configuration

The `optuna_best` model uses:

| Parameter | Value |
|-----------|-------|
| Architecture | ConvNeXt Tiny |
| Classification | 120 bins (3° per bin) |
| Label Smoothing | 21.9° Gaussian |
| Input Size | 224×224 RGB |

The backend automatically reads these from your model's checkpoint or `config.json`.

---

## Security Features

| Feature | Protection |
|---------|------------|
| API Key | Authenticates all requests |
| Rate Limiting | 10 requests/hour per client |
| File Size Limit | 200 MB max upload |
| Duration Limit | 2 minutes max processing |
| FPS Limit | 5-15 FPS (prevents abuse) |

---

## Troubleshooting

### Modal deployment fails with encoding error (Windows)

```powershell
$env:PYTHONUTF8="1"; modal deploy modal_app.py
```

### "API key required" error

1. Check `.env.local` has correct `MODAL_API_KEY`
2. Verify secret exists: `modal secret list`
3. Redeploy backend: `modal deploy modal_app.py`

### Video not loading in browser

1. Check browser console (F12) for errors
2. Verify Modal endpoints are running: `modal app list`
3. Test health endpoint: `curl YOUR-MODAL-URL/health`

### Processing is slow

- First request after idle takes ~30s (Modal cold start)
- Reduce FPS to 5 for faster processing
- Select shorter video clips

---

## Development

### Test Backend Locally

```bash
cd backend
modal serve modal_app.py  # Hot-reload enabled
```

### View Modal Logs

```bash
modal app logs bikefitting
```

### Check Volume Contents

```bash
modal volume ls bikefitting-models
modal volume ls bikefitting-temp
```
