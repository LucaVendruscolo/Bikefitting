import { NextRequest, NextResponse } from 'next/server';

const GITHUB_RELEASE_URL = 'https://github.com/LucaVendruscolo/Bikefitting/releases/download/models-v1';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  const filePath = params.path.join('/');
  const githubUrl = `${GITHUB_RELEASE_URL}/${filePath}`;
  
  console.log(`[Model Proxy] Fetching: ${githubUrl}`);
  
  try {
    const response = await fetch(githubUrl, {
      headers: {
        'User-Agent': 'BikeFit-Pro/1.0',
      },
    });
    
    if (!response.ok) {
      console.error(`[Model Proxy] Failed to fetch ${filePath}: ${response.status}`);
      return NextResponse.json(
        { error: `Failed to fetch model: ${response.statusText}` },
        { status: response.status }
      );
    }
    
    const contentType = filePath.endsWith('.json') 
      ? 'application/json' 
      : 'application/octet-stream';
    
    const data = await response.arrayBuffer();
    
    return new NextResponse(data, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Content-Length': data.byteLength.toString(),
        'Cache-Control': 'public, max-age=31536000, immutable',
        'Access-Control-Allow-Origin': '*',
      },
    });
  } catch (error) {
    console.error(`[Model Proxy] Error fetching ${filePath}:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch model' },
      { status: 500 }
    );
  }
}

