module.exports = {
  publicPath: process.env.NODE_ENV === 'production' ? './' : '/',
  outputDir: 'dist',
  assetsDir: 'static',
  productionSourceMap: false,
  devServer: {
    port: 8080,
    open: true,
    client: {
      overlay: {
        warnings: false,
        errors: true
      }
    },
    proxy: {
      '/api': {
        target: 'http://10.101.64.214:5000',
        changeOrigin: true,
        pathRewrite: {
          '^/api': ''
        }
      }
    }
  }
} 