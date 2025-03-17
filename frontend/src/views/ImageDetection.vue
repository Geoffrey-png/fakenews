<template>
  <div class="image-detection">
    <div class="page-header">
      <h1>图片内容检测</h1>
      <p>在下方上传图片，系统将分析其真实性</p>
    </div>

    <el-card class="main-card">
      <!-- 原有上传组件 -->
      <el-upload
        class="upload-container"
        :action="uploadUrl"
        :show-file-list="false"
        :before-upload="beforeUpload"
        :on-success="handleSuccess"
        :on-error="handleError"
      >
        <div class="upload-area">
          <el-button type="primary" size="medium">点击上传图片</el-button>
          <div class="el-upload__tip">支持JPG/PNG格式，大小不超过5MB</div>
        </div>
      </el-upload>

      <!-- 图片预览 -->
      <div v-if="imageUrl" class="preview-area">
        <el-image 
          :src="imageUrl"
          fit="contain"
          style="max-height: 400px;"
        >
          <div slot="error" class="image-slot">
            <i class="el-icon-picture-outline"></i>
          </div>
        </el-image>
      </div>

      <!-- 检测结果 -->
      <div v-if="result" class="result-container">
        <el-alert
          :title="result.title"
          :type="result.type"
          :description="result.description"
          show-icon
          class="result-alert"
        />
        <el-button 
          type="info" 
          @click="resetForm"
          class="reset-btn"
        >
          重新检测
        </el-button>
      </div>

      <!-- 加载状态 -->
      <div v-if="loading" class="loading-container">
        <el-progress 
          :percentage="progress" 
          status="success" 
          :show-text="false"
        />
        <p>正在分析图片内容，请稍候...</p>
      </div>
    </el-card>
  </div>
</template>

<style scoped>
.image-detection {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}

.page-header {
  text-align: center;
  margin-bottom: 40px;
}

.page-header h1 {
  color: #409EFF;
  font-size: 28px;
  margin-bottom: 15px;
}

.page-header p {
  color: #606266;
  font-size: 16px;
}

/* 保持原有上传组件样式不变 */
.upload-container {
  margin: 50px auto;
  text-align: center;
  padding: 30px;
  border: 2px dashed #eee;
}

.preview-area {
  margin: 30px 0;
  text-align: center;
}

.result-alert {
  margin: 30px auto;
  max-width: 600px;
}

.reset-btn {
  margin-top: 20px;
  width: 200px;
}

.loading-container {
  margin: 40px 0;
  text-align: center;
}
</style>

<script>
export default {
  name: 'ImageDetection',
  data() {
    return {
      imageUrl: '',
      result: null,
      loading: false,
      uploadUrl: process.env.VUE_APP_API_BASE + '/upload-image' // 添加API地址配置
    }
  },
  methods: {
    beforeUpload(file) {
      const isImage = ['image/jpeg', 'image/png'].includes(file.type)
      const isLt5M = file.size / 1024 / 1024 < 5

      if (!isImage) {
        this.$message.error('只能上传 JPG/PNG 格式的图片!')
        return false // 明确返回false阻止上传
      }
      if (!isLt5M) {
        this.$message.error('图片大小不能超过5MB!')
        return false // 明确返回false阻止上传
      }
      return true
    },

    handleSuccess(response, file) { // 添加第二个参数接收file对象
      this.imageUrl = URL.createObjectURL(file.raw)
      this.loading = true
      
      // 添加错误处理
      this.$api.uploadImage(file.raw)
        .then(res => {
          if (!res.data.confidence) {
            throw new Error('无效的API响应')
          }
          this.result = {
            title: res.data.is_fake ? '疑似虚假新闻' : '可信新闻',
            type: res.data.is_fake ? 'error' : 'success',
            description: `可信度评分: ${(res.data.confidence * 100).toFixed(1)}%`
          }
        })
        .catch(error => {
          this.$message.error(`检测失败: ${error.message}`)
          this.result = null
        })
        .finally(() => {
          this.loading = false
        })
    }
  },
  beforeDestroy() {
    if (this.imageUrl) {
      URL.revokeObjectURL(this.imageUrl) // 释放对象URL
    }
  }
}
</script>