<template>
  <div class="loading-container" v-if="loading">
    <div class="loading-overlay">
      <div class="loading-content">
        <el-progress type="circle" :percentage="percentage" :status="status"></el-progress>
        <p class="loading-text">{{ text }}</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'LoadingIndicator',
  props: {
    loading: {
      type: Boolean,
      default: false
    },
    text: {
      type: String,
      default: '加载中，请稍候...'
    },
    percentage: {
      type: Number,
      default: 100
    },
    status: {
      type: String,
      default: ''
    }
  }
}
</script>

<style scoped>
.loading-container {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 9999;
}

.loading-overlay {
  position: absolute;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.loading-content {
  background-color: white;
  padding: 30px;
  border-radius: 8px;
  text-align: center;
  box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
}

.loading-text {
  margin-top: 15px;
  color: #606266;
  font-size: 16px;
}
</style> 