import Vue from 'vue'
import App from './App.vue'
import router from './router'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import './assets/styles.css'
import apiService from './utils/api'
import axios from 'axios'

// 创建axios实例
const http = axios.create({
  baseURL: process.env.VUE_APP_API_BASE || '',
  timeout: 30000 // 30秒超时
})

// 请求拦截器
http.interceptors.request.use(
  config => {
    console.log('发送请求到:', config.url)
    return config
  },
  error => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
http.interceptors.response.use(
  response => {
    console.log('接收响应:', response.config.url)
    return response
  },
  error => {
    console.error('响应错误:', error.message)
    return Promise.reject(error)
  }
)

// 注册全局API服务
Vue.prototype.$api = apiService
// 注册全局HTTP客户端
Vue.prototype.$http = http

// 使用Element UI
Vue.use(ElementUI, {
  size: 'medium' // 设置组件默认尺寸
})

Vue.config.productionTip = false

// 全局错误处理
Vue.config.errorHandler = (err, vm, info) => {
  console.error('Vue错误:', err)
  console.error('错误信息:', info)
}

new Vue({
  router,
  render: h => h(App)
}).$mount('#app') 