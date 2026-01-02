/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Enable static export for Vercel
  output: 'export',
  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
  // Configure webpack for ONNX Runtime
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };
    return config;
  },
};

module.exports = nextConfig;
