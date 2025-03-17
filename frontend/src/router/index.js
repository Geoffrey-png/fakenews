import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '../views/Home.vue'
import SingleDetection from '../views/SingleDetection.vue'
import BatchDetection from '../views/BatchDetection.vue'
import About from '../views/About.vue'
import VideoDetection from '../views/VideoDetection.vue'  // 导入VideoDetection组件
import ImageDetection from '../views/ImageDetection.vue'  // 导入ImageDetection组件
import HybridDetection from '../views/HybridDetection.vue'  // 导入HybridDetection组件
Vue.use(VueRouter)

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/single',
    name: 'SingleDetection',
    component: SingleDetection
  },
  {
    path: '/batch',
    name: 'BatchDetection',
    component: BatchDetection
  },
  {
    path: '/about',
    name: 'About',
    component: About
  },
  {
    path: '/video-detection',
    name: 'video-detection',
    component: VideoDetection  // 修正为已导入的组件
  },
  {
    path: '/image-detection',
    name: 'ImageDetection',
    component: ImageDetection  // 修正为已导入的组件
  },
  {
    path: '/hybrid-detection',
    name: 'HybridDetection',
    component: HybridDetection  // 修正为已导入的组件
  },
  // 添加通配符路由，处理404情况
  {
    path: '*',
    redirect: '/'
  }
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router 