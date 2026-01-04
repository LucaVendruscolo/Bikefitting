# Bikefitting Web App

Upload a cycling video and get AI analysis of your riding position.

## How it works

1. You upload a video through the website
2. The video gets sent to a cloud GPU (Modal) for processing
3. AI models detect the bike and your body position
4. You see the results with angle measurements overlaid

## What you need

- Node.js 18 or newer
- A Modal account (modal.com)
- The best_model.pt file uploaded to Modal

## Setup the backend (Modal)

First, install Modal and log in:
```
pip install modal
modal setup
```

Create storage for the model:
```
modal volume create bikefitting-models
modal volume put bikefitting-models ../models/best_model.pt best_model.pt
```

Create a temp storage for processed videos:
```
modal volume create bikefitting-temp
```

Deploy the backend:
```
cd backend
modal deploy modal_app.py
```

After deploying, Modal will give you URLs. You need the ones for:
- process_video_stream
- download

## Setup the frontend

```
cd frontend
npm install
```

Create a file called .env.local with your Modal URLs:
```
NEXT_PUBLIC_MODAL_STREAM_URL=https://YOUR-USERNAME--bikefitting-process-video-stream.modal.run
NEXT_PUBLIC_MODAL_DOWNLOAD_URL=https://YOUR-USERNAME--bikefitting-download.modal.run
```

Replace YOUR-USERNAME with your Modal username.

Run the dev server:
```
npm run dev
```

Open http://localhost:3000

## Deploy to the internet (Vercel)

```
npm install -g vercel
cd frontend
vercel
```

When asked, add your environment variables in the Vercel dashboard:
- NEXT_PUBLIC_MODAL_STREAM_URL
- NEXT_PUBLIC_MODAL_DOWNLOAD_URL

## Limits

- Max video length: 2 minutes
- Max file size: 200 MB
- Output FPS: 5-15 (lower = faster processing)

## Troubleshooting

Video won't load?
- Check your browser console for errors
- Make sure Modal is deployed and running
- Check your .env.local URLs are correct

Processing is slow?
- Use a lower FPS (5 is fastest)
- Select a shorter time range
- First request takes ~30 seconds for Modal to start up
