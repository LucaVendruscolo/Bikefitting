/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  
  // Configure webpack for ONNX Runtime Web
  webpack: (config, { isServer }) => {
    // Don't bundle ONNX node bindings on client
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
        perf_hooks: false,
      };
      
      // Exclude node-specific ONNX files
      config.externals = config.externals || [];
      config.externals.push({
        'onnxruntime-node': 'commonjs onnxruntime-node',
      });
    }
    
    // Handle .wasm files
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };
    
    return config;
  },
  
  // Disable server-side features that cause issues
  experimental: {
    serverComponentsExternalPackages: ['onnxruntime-web'],
  },
};

module.exports = nextConfig;
