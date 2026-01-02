/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static export for Vercel
  output: 'export',
  
  // Required for ONNX Runtime Web
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
    };
    return config;
  },
  
  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
};

module.exports = nextConfig;

