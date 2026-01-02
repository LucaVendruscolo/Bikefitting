/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable static export for Vercel
  output: 'export',
  
  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
  
  // Configure webpack for ONNX Runtime Web
  webpack: (config, { isServer }) => {
    // Don't bundle ONNX Runtime on server side
    if (isServer) {
      config.externals = config.externals || [];
      config.externals.push('onnxruntime-web');
    }
    
    // Handle ONNX Runtime Web properly
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false,
      perf_hooks: false,
    };
    
    // Exclude problematic ONNX node files from processing
    config.module.rules.push({
      test: /ort.*\.node\..*\.mjs$/,
      type: 'javascript/auto',
      resolve: {
        fullySpecified: false,
      },
    });
    
    // Ignore node-specific ONNX files
    config.plugins.push(
      new (require('webpack').IgnorePlugin)({
        resourceRegExp: /^onnxruntime-node$/,
      })
    );
    
    return config;
  },
  
  // Transpile ONNX Runtime
  transpilePackages: ['onnxruntime-web'],
};

module.exports = nextConfig;
