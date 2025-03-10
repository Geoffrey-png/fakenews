import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '../views/Home.vue'
import SingleDetection from '../views/SingleDetection.vue'
import BatchDetection from '../views/BatchDetection.vue'
import About from '../views/About.vue'

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