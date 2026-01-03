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
}

module.exports = nextConfig

