import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '../views/Home.vue'
import SingleDetection from '../views/SingleDetection.vue'
import BatchDetection from '../views/BatchDetection.vue'
import About from '../views/About.vue'
import VideoDetection from '../views/VideoDetection.vue'  // 新增组件导入
import ImageDetection from '../views/ImageDetection.vue'
import HybridDetection from '../views/HybridDetection.vue'

Vue.use(VueRouter)

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/ABOUT',
    name: 'about',
    component: About
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
  }
]

const router = new VueRouter({
  mode: 'history',
  base: process.env.BASE_URL,
  routes
})

export default router 