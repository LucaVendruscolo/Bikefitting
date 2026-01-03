/** @type {import('next').NextConfig} */
const nextConfig = {
  // Configure allowed domains for images/videos if needed
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**.modal.run',
      },
    ],
  },
  // Increase body size limit for video uploads (50MB base64 ≈ 67MB JSON)
  experimental: {
    serverActions: {
      bodySizeLimit: '100mb',
    },
  },
  // Security headers
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
    ]
  },
}

module.exports = nextConfig

