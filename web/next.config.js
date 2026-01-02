/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export',
  images: {
    unoptimized: true,
  },
  // ONNX Runtime is loaded from CDN, so we don't need special webpack config
};

module.exports = nextConfig;
