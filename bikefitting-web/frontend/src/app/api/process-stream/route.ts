import { NextRequest } from 'next/server'

export const maxDuration = 300 // 5 minutes

const MAX_DURATION_SEC = 120
const MAX_OUTPUT_FPS = 15
const MIN_OUTPUT_FPS = 5

/**
 * Streaming API route - proxies SSE from Modal backend for real-time progress.
 * 
 * Request: { video_base64, output_fps, start_time, end_time, max_duration_sec }
 * Response: SSE stream with progress events, then 'complete' event with results
 */
export async function POST(request: NextRequest) {
  const body = await request.json()
  const { video_base64, output_fps = 10, start_time = 0, end_time = 0, max_duration_sec = 60 } = body

  if (!video_base64) {
    return new Response(
      `data: ${JSON.stringify({ type: 'error', error: 'video_base64 is required' })}\n\n`,
      { status: 400, headers: { 'Content-Type': 'text/event-stream' } }
    )
  }

  // Enforce limits
  const safeFps = Math.max(MIN_OUTPUT_FPS, Math.min(MAX_OUTPUT_FPS, Number(output_fps) || 10))
  const safeDuration = Math.min(MAX_DURATION_SEC, Number(max_duration_sec) || 60)
  const safeStartTime = Math.max(0, Number(start_time) || 0)
  const safeEndTime = Number(end_time) || 0

  const modalUrl = process.env.MODAL_STREAM_URL
  if (!modalUrl) {
    return new Response(
      `data: ${JSON.stringify({ type: 'error', error: 'MODAL_STREAM_URL not configured' })}\n\n`,
      { status: 500, headers: { 'Content-Type': 'text/event-stream' } }
    )
  }
  const apiKey = process.env.MODAL_API_KEY

  try {
    const response = await fetch(modalUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_base64,
        api_key: apiKey,
        output_fps: safeFps,
        start_time: safeStartTime,
        end_time: safeEndTime,
        max_duration_sec: safeDuration,
      }),
    })

    if (!response.ok || !response.body) {
      const text = await response.text()
      return new Response(
        `data: ${JSON.stringify({ type: 'error', error: text || 'Server connection failed' })}\n\n`,
        { status: response.status, headers: { 'Content-Type': 'text/event-stream' } }
      )
    }

    // Stream response through to client
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })
  } catch (error) {
    console.error('Stream error:', error)
    return new Response(
      `data: ${JSON.stringify({ type: 'error', error: 'Processing request failed' })}\n\n`,
      { status: 500, headers: { 'Content-Type': 'text/event-stream' } }
    )
  }
}
