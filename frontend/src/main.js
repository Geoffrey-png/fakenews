import Vue from 'vue'
import App from './App.vue'
import router from './router'
import ElementUI from 'element-ui'
import 'element-ui/lib/theme-chalk/index.css'
import './assets/styles.css'
import apiService from './utils/api'

// 注册全局API服务
Vue.prototype.$api = apiService

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