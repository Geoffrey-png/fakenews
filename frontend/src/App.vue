<template>
  <div id="app">
    <el-container>
      <el-header>
        <div class="header-content">
          <div class="logo">
            <img src="./assets/logo.png" alt="Logo" v-if="false">
            <h1>面向新闻场景的伪造检测平台</h1>
          </div>
          <el-menu mode="horizontal" router :default-active="activeRoute" class="nav-menu">
            <el-menu-item index="/">首页</el-menu-item>
            <el-menu-item index="/single">单文本检测</el-menu-item>
            <el-menu-item index="/batch">批量检测</el-menu-item>
            <el-menu-item index="/image-detection">图片检测</el-menu-item>
            <el-menu-item index="/hybrid-detection">混合检测</el-menu-item>  <!-- 新增混合检测选项 -->
             <el-menu-item index="/video-detection">视频检测</el-menu-item>  <!-- 新增视频检测入口 -->
            <el-menu-item index="/about" style="margin-left: auto">关于系统</el-menu-item> 
          </el-menu>
        </div>
      </el-header>
      <el-main>
        <router-view/>
      </el-main>
      <el-footer>
        <p>© 2025Clear Fake - 基于深度学习的新闻真伪判别平台</p>
      </el-footer>
    </el-container>
  </div>
</template>

<script>
export default {
  name: 'App',
  computed: {
    activeRoute() {
      return this.$route.path;
    }
  }
}
</script>

<style>
#app {
  font-family: 'Microsoft YaHei', 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  color: #2c3e50;
  min-height: 100vh;
}

body {
  margin: 0;
  padding: 0;
}

.el-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.el-header {
  background-color: #409EFF;
  color: white;
  height: auto !important;
  padding: 0;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 20px;
}

.logo {
  display: flex;
  align-items: center;
}

.logo img {
  height: 40px;
  margin-right: 10px;
}

.logo h1 {
  font-size: 1.5rem;
  margin: 0;
  white-space: nowrap;
}

.nav-menu {
  background-color: transparent;
  border-bottom: none;
}

.el-menu--horizontal > .el-menu-item {
  color: #fff;
  border-bottom: 2px solid transparent !important;
  /* 新增悬停动画 */
  transition: border-color 0.3s ease-in-out;
}

/* 新增悬停效果 */
.el-menu--horizontal > .el-menu-item:hover {
  border-bottom: 2px solid #fff !important;
  background-color: rgba(255, 255, 255, 0.2);
}

/* 保持原有激活状态样式不变 */
.el-menu--horizontal > .el-menu-item.is-active {
  color: #fff;
  border-bottom: 2px solid #409EFF !important;  /* 将#fff改为主题蓝色 */
  background-color: rgba(255, 255, 255, 0.1);
}

.el-main {
  background-color: #f5f7fa;
  flex: 1;
  padding: 20px;
}

.el-footer {
  background-color: #545c64;
  color: #eee;
  text-align: center;
  line-height: 60px;
  padding: 0;
}
</style> 