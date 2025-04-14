const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  publicPath: process.env.NODE_ENV === 'production' ? './' : '/',
  outputDir: 'dist',
  assetsDir: 'static',
  productionSourceMap: false,
  transpileDependencies: true,
  devServer: {
    port: 8080,
    open: true,
    client: {
      overlay: {
        warnings: false,
        errors: true
      },
      maxWebsocketSize: 524288000, // 500MB in bytes
    },
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        ws: true,
        pathRewrite: {
          '^/api': ''
        }
      }
    }
  },
  // 增加webpack配置以支持大文件
  configureWebpack: {
    performance: {
      maxAssetSize: 524288000,
      maxEntrypointSize: 524288000,
    }
  }
})