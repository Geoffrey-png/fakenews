<template>
  <div class="video-detection">
    <el-card class="main-card">
      <h2 class="title">视频真实性检测</h2>
      
      <!-- 视频上传区域 -->
      <el-upload
        class="upload-container"
        action=""
        :auto-upload="false"
        :on-change="handleVideoUpload"
        :show-file-list="false"
        accept="video/mp4,video/avi,video/quicktime"
      >
        <div class="upload-area">
          <div class="upload-content">
            <i class="el-icon-upload" style="font-size: 40px; margin-bottom: 10px;"/>
            <div class="upload-text">
              <p>点击上传视频文件（支持MP4/AVI/MOV格式）</p>
              <p class="tip-text">最大支持500MB的视频文件</p>
            </div>
          </div>
        </div>
      </el-upload>
      
      <!-- 新增检测按钮 -->
      <div class="upload-action" v-if="uploadedVideo">
        <el-button 
          type="primary" 
          :loading="loading"
          @click="startDetection"
          icon="el-icon-video-play"
        >
          {{ loading ? '分析中...' : '立即检测' }}
        </el-button>
      </div>
      <!-- 视频预览 -->
      <div v-if="uploadedVideo" class="video-preview">
        <video 
          controls
          :src="videoPreviewUrl"
          class="preview-player"
        />
        <el-button 
          type="primary" 
          :loading="loading"
          @click="startDetection"
        >
          {{ loading ? '分析中...' : '开始检测' }}
        </el-button>
      </div>

      <!-- 检测结果展示 -->
      <div v-if="results" class="result-container">
        <h3>检测结果</h3>
        <el-alert
          :title="results.status === 'real' ? '真实视频' : '疑似伪造视频'"
          :type="results.status === 'real' ? 'success' : 'error'"
          :closable="false"
        />
        <el-row :gutter="20" class="result-details">
          <el-col :span="8">
            <div class="detail-item">
              <label>置信度：</label>
              <span>{{ (results.confidence * 100).toFixed(1) }}%</span>
            </div>
          </el-col>
          <el-col :span="16">
            <div class="detail-item">
              <label>分析结果：</label>
              <span>{{ results.description }}</span>
            </div>
          </el-col>
        </el-row>
        <el-button 
          type="info" 
          @click="resetForm"
          class="reset-btn"
        >
          重新检测
        </el-button>
      </div>
    </el-card>
  </div>
</template>

<script>
export default {
  name: 'VideoDetection',
  data() {
    return {
      uploadedVideo: null,
      videoPreviewUrl: '',
      results: null,
      loading: false,
      maxSize: 500 // MB
    }
  },
  methods: {
    handleVideoUpload(file) {
      if (file.size > this.maxSize * 1024 * 1024) {
        this.$message.error(`文件大小不能超过${this.maxSize}MB`)
        return false
      }
      this.uploadedVideo = file.raw
      this.videoPreviewUrl = URL.createObjectURL(file.raw)
    },
    
    async startDetection() {
      if (!this.uploadedVideo) {
        this.$message.warning('请先上传视频文件')
        return
      }
      
      this.loading = true
      try {
        const formData = new FormData()
        formData.append('video', this.uploadedVideo)
        
        // 调用视频检测API
        const response = await this.$api.uploadVideo(formData)
        this.results = response.data
        this.$message.success('视频分析完成')
      } catch (error) {
        console.error('检测失败:', error)
        this.$message.error('视频分析失败，请稍后重试')
      } finally {
        this.loading = false
      }
    },
    
    resetForm() {
      this.uploadedVideo = null
      this.videoPreviewUrl = ''
      this.results = null
    }
  }
}
</script>

<style scoped>
.title {
  text-align: center;
  color: #409EFF;
  margin-bottom: 30px;
}

.upload-container {
  border: 2px dashed #ddd;
  border-radius: 6px;
  padding: 20px;
  margin: 0 auto 30px; /* 增加水平居中 */
  display: flex; /* 新增flex布局 */
  justify-content: center; /* 新增水平居中 */
  width: 80%; /* 限制宽度增强居中效果 */
}

.upload-area {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center; /* 新增垂直居中 */
  min-height: 200px; /* 设置最小高度保证垂直居中效果 */
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.upload-text {
  text-align: center;
  margin-top: 10px;
}

.tip-text {
  color: #909399;
  font-size: 0.9em;
}

.video-preview {
  text-align: center;
  margin-bottom: 30px;
}

.preview-player {
  width: 100%;
  max-width: 600px;
  margin-bottom: 20px;
  border-radius: 4px;
}

.result-container {
  padding: 20px;
  background: #f5f7fa;
  border-radius: 4px;
}

.result-details {
  margin: 20px 0;
}

.detail-item {
  background: white;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 10px;
}

.reset-btn {
  margin-top: 20px;
  width: 100%;
}
</style>

// 在api.js中需要添加视频检测方法
const apiService = {
  uploadVideo(formData) {
    return api.post('/video-analysis', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  }
}