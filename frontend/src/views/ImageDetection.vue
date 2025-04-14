<template>
  <div class="image-detection">
    <div class="page-header">
      <h1>图片内容检测</h1>
      <p>在下方上传图片，系统将使用Hammer检测技术分析图片真实性</p>
    </div>

    <el-card class="main-card">
      <!-- 上传组件 -->
      <el-upload
        class="upload-container"
        action="#"
        :auto-upload="true"
        :show-file-list="false"
        :before-upload="handleBeforeUpload"
        :http-request="customUpload"
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
          show-icon
          class="result-alert"
        >
          <div class="result-details">
            <p><strong>可信度评分:</strong> {{ result.confidence.toFixed(1) }}%</p>
            <p v-if="result.manipulationType"><strong>篡改类型:</strong> {{ result.manipulationType }}</p>
          </div>
        </el-alert>

        <!-- 篡改区域可视化图像 -->
        <div v-if="result.visualizationUrl" class="visualization-area">
          <h3>篡改区域检测结果</h3>
          <el-image 
            :src="result.visualizationUrl"
            fit="contain"
            style="max-height: 400px; border: 1px solid #ddd;"
          >
            <div slot="error" class="image-slot">
              <i class="el-icon-picture-outline"></i>
              <p>检测结果图像加载失败</p>
            </div>
          </el-image>
        </div>
        
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
        />
        <p>正在使用Hammer技术分析图片内容，请稍候...</p>
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
  max-width: 800px;
}

.result-details {
  margin-top: 15px;
  padding: 10px;
  background-color: #f9f9f9;
  border-radius: 4px;
}

.visualization-area {
  margin: 30px auto;
  text-align: center;
  max-width: 800px;
}

.visualization-area h3 {
  margin-bottom: 15px;
  color: #606266;
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
      imageFile: null,
      result: null,
      loading: false,
      progress: 0,
      progressTimer: null
    }
  },
  computed: {
    apiBaseUrl() {
      console.log('API基础URL:', process.env.VUE_APP_API_BASE || '');
      return process.env.VUE_APP_API_BASE || '';
    }
  },
  methods: {
    // 上传前的验证
    handleBeforeUpload(file) {
      console.log('准备上传文件:', file.name);
      const isImage = ['image/jpeg', 'image/png'].includes(file.type)
      const isLt5M = file.size / 1024 / 1024 < 5

      if (!isImage) {
        this.$message.error('只能上传 JPG/PNG 格式的图片!')
        return false
      }
      if (!isLt5M) {
        this.$message.error('图片大小不能超过5MB!')
        return false
      }
      
      // 保存文件并显示预览
      this.imageFile = file
      this.imageUrl = URL.createObjectURL(file)
      console.log('文件预览URL:', this.imageUrl);
      
      return true
    },
    
    // 自定义上传方法
    customUpload(options) {
      console.log('开始自定义上传:', options.file.name);
      this.startDetection(options.file)
    },
    
    // 开始检测
    startDetection(file = null) {
      // 如果提供了文件参数，则使用它
      if (file) {
        this.imageFile = file
      }
      
      if (!this.imageFile) {
        this.$message.warning('请先上传图片')
        return
      }
      
      console.log('开始检测文件:', this.imageFile.name);
      this.loading = true
      this.result = null
      this.startProgressSimulation()
      
      // 创建表单数据
      const formData = new FormData()
      formData.append('image', this.imageFile)
      
      // 直接使用完整URL，不使用拼接
      const url = 'http://10.101.64.214:5000/detect/image';
      console.log('发送请求到:', url);
      
      // 输出环境变量和配置信息
      console.log('环境变量:', process.env);
      console.log('API基础URL:', this.apiBaseUrl);
      
      // 调用API
      this.$http.post(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })
      .then(response => {
        console.log('收到API响应:', response.data);
        const data = response.data
        
        if (data.success) {
          // 处理检测结果
          this.result = {
            title: data.is_fake ? '检测到图像篡改' : '未检测到篡改',
            type: data.is_fake ? 'warning' : 'success',
            confidence: data.confidence,
            manipulationType: data.manipulation_type,
            fakeRegion: data.fake_region,
            visualizationUrl: data.visualization_url 
              ? `http://10.101.64.214:5000${data.visualization_url}` 
              : null
          }
          console.log('处理后的结果:', this.result);
        } else {
          throw new Error(data.error || '检测失败')
        }
      })
      .catch(error => {
        console.error('检测失败详情:', error);
        if (error.response) {
          console.error('错误响应状态:', error.response.status);
          console.error('错误响应数据:', error.response.data);
        }
        this.$message.error(`检测失败: ${error.message || '未知错误'}`);
      })
      .finally(() => {
        this.loading = false
        this.stopProgressSimulation()
      })
    },
    
    // 重置表单
    resetForm() {
      if (this.imageUrl) {
        URL.revokeObjectURL(this.imageUrl)
      }
      this.imageUrl = ''
      this.imageFile = null
      this.result = null
    },
    
    // 模拟进度条
    startProgressSimulation() {
      this.progress = 0
      this.progressTimer = setInterval(() => {
        if (this.progress < 90) {
          this.progress += Math.random() * 10
          if (this.progress > 90) this.progress = 90
        }
      }, 500)
    },
    
    stopProgressSimulation() {
      clearInterval(this.progressTimer)
      this.progress = 100
    }
  },
  beforeDestroy() {
    if (this.imageUrl) {
      URL.revokeObjectURL(this.imageUrl)
    }
    if (this.progressTimer) {
      clearInterval(this.progressTimer)
    }
  }
}
</script>