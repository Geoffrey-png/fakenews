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
      
      <!-- 视频预览 -->
      <div v-if="uploadedVideo" class="video-preview">
        <video 
          controls
          :src="videoPreviewUrl"
          class="preview-player"
        />
        <div class="button-container">
          <el-button 
            type="primary" 
            :loading="loading"
            @click="startDetection"
            size="large"
          >
            {{ loading ? '分析中...' : '开始检测' }}
          </el-button>
        </div>
      </div>

      <!-- 处理进度条 -->
      <el-progress 
        v-if="loading"
        :percentage="processingProgress" 
        :format="progressFormat"
        style="margin-top: 20px;"
      />

      <!-- 检测结果展示 -->
      <div v-if="results" class="result-container">
        <h3>检测结果</h3>
        
        <!-- 处理未找到图片的情况 -->
        <el-alert
          v-if="!results.success"
          title="图片加载失败"
          type="warning"
          :description="results.error"
          :closable="false"
          show-icon
          style="margin-bottom: 20px;"
        />
        
        <!-- 处理成功的情况 -->
        <template v-else>
          <el-alert
            :title="hasFake ? '疑似伪造视频' : '未检测到明显伪造'"
            :type="hasFake ? 'error' : 'success'"
            :closable="false"
            style="margin-bottom: 20px;"
          />
          
          <div class="summary-box">
            <h4>分析总结</h4>
            <p>{{ results.summary }}</p>
            <p class="processing-time">处理时间: {{ results.processing_time.toFixed(2) }}秒</p>
          </div>
          
          <!-- 视频帧分析结果 -->
          <h4 class="frames-title">视频帧分析 (共{{ results.total_frames }}帧)</h4>
          
          <div class="frames-container">
            <div 
              v-for="(frame, index) in results.frames" 
              :key="index"
              class="frame-item"
              :class="{'fake-frame': frame.is_fake}"
            >
              <div class="frame-image">
                <img :src="frame.frame_path" :alt="`Frame ${frame.frame_number}`" />
                <div v-if="frame.is_fake" class="fake-marker">伪造</div>
              </div>
              <div class="frame-info">
                <div class="frame-number">帧 #{{ frame.frame_number }}</div>
                <div class="frame-result" :class="frame.is_fake ? 'fake' : 'real'">
                  {{ frame.is_fake ? '伪造' : '真实' }}
                  <span class="confidence">({{ (frame.confidence * 100).toFixed(1) }}%)</span>
                </div>
                <div v-if="frame.is_fake" class="manipulation-type">
                  {{ frame.manipulation_type }}
                </div>
              </div>
            </div>
          </div>
        </template>
        
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
      processingProgress: 0,
      progressInterval: null,
      maxSize: 500 // MB
    }
  },
  computed: {
    hasFake() {
      if (!this.results || !this.results.frames) return false;
      return this.results.frames.some(frame => frame.is_fake);
    }
  },
  methods: {
    handleVideoUpload(file) {
      // 检查文件大小
      if (file.size > this.maxSize * 1024 * 1024) {
        this.$message.error(`文件大小不能超过${this.maxSize}MB`)
        return false
      }
      
      // 检查文件类型
      const acceptedTypes = ['video/mp4', 'video/avi', 'video/quicktime']
      if (!acceptedTypes.includes(file.raw.type)) {
        this.$message.error('请上传MP4、AVI或MOV格式的视频文件')
        return false
      }

      // 创建新的视频预览URL之前，先清理旧的URL
      if (this.videoPreviewUrl) {
        URL.revokeObjectURL(this.videoPreviewUrl)
      }

      // 保存文件并创建预览URL
      this.uploadedVideo = file.raw
      this.videoPreviewUrl = URL.createObjectURL(this.uploadedVideo)

      // 返回false阻止自动上传
      return false
    },
    
    progressFormat(percentage) {
      return percentage < 100 ? '处理中...' : '完成';
    },
    
    async startDetection() {
      if (!this.uploadedVideo) {
        this.$message.warning('请先上传视频文件')
        return
      }
      
      this.loading = true
      this.processingProgress = 0
      
      // 模拟进度更新
      this.progressInterval = setInterval(() => {
        if (this.processingProgress < 90) {
          this.processingProgress += Math.random() * 5;
        }
      }, 500);
      
      try {
        // 调用视频分析API
        const response = await this.$api.analyzeVideo(this.uploadedVideo);
        this.results = response.data;
        this.$message.success('视频分析完成');
        this.processingProgress = 100;
      } catch (error) {
        console.error('检测失败:', error);
        this.$message.error('视频分析失败，请稍后重试');
      } finally {
        clearInterval(this.progressInterval);
        this.loading = false;
      }
    },
    
    resetForm() {
      this.uploadedVideo = null;
      this.videoPreviewUrl = '';
      this.results = null;
      this.processingProgress = 0;
    }
  },
  beforeDestroy() {
    // 清理定时器
    if (this.progressInterval) {
      clearInterval(this.progressInterval);
    }
    
    // 释放视频URL对象
    if (this.videoPreviewUrl) {
      URL.revokeObjectURL(this.videoPreviewUrl);
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

.button-container {
  display: flex;
  justify-content: center;
  margin-top: 15px;
}

.preview-player {
  width: 100%;
  max-width: 600px;
  margin-bottom: 10px;
  border-radius: 4px;
}

.result-container {
  padding: 20px;
  background: #f5f7fa;
  border-radius: 4px;
}

.summary-box {
  background: white;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 20px;
}

.processing-time {
  color: #909399;
  font-size: 0.9em;
  margin-top: 10px;
}

.frames-title {
  margin: 20px 0 15px;
}

.frames-container {
  display: flex;
  flex-wrap: wrap;
  gap: 15px;
  margin-bottom: 20px;
}

.frame-item {
  width: calc(25% - 15px);
  background: white;
  border-radius: 4px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s;
  position: relative;
}

.frame-item:hover {
  transform: translateY(-5px);
}

.fake-frame {
  border: 2px solid #F56C6C;
}

.frame-image {
  position: relative;
  width: 100%;
  height: 150px;
  overflow: hidden;
}

.frame-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.fake-marker {
  position: absolute;
  top: 5px;
  right: 5px;
  background: rgba(245, 108, 108, 0.9);
  color: white;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 0.8em;
}

.frame-info {
  padding: 10px;
}

.frame-number {
  font-weight: bold;
  margin-bottom: 5px;
}

.frame-result {
  margin-bottom: 5px;
}

.frame-result.fake {
  color: #F56C6C;
  font-weight: bold;
}

.frame-result.real {
  color: #67C23A;
}

.confidence {
  font-size: 0.9em;
  opacity: 0.8;
}

.manipulation-type {
  font-size: 0.9em;
  color: #409EFF;
}

.reset-btn {
  margin-top: 20px;
  width: 100%;
}

@media (max-width: 768px) {
  .frame-item {
    width: calc(50% - 10px);
  }
}

@media (max-width: 480px) {
  .frame-item {
    width: 100%;
  }
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
  },
  analyzeVideo(video) {
    return api.post('/video-analysis/analyze', { video }, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  }
}