<template>
  <div class="hybrid-detection">
    <div class="page-header">
      <h1>混合内容检测</h1>
      <p>输入需要检测的内容URL，系统将分析其真实性</p>
    </div>

    <el-card class="main-card">
      <h2 class="page-title">混合内容检测</h2>
      
      <!-- 新增默认状态提示 -->
      <el-alert
        v-if="!results && !inputUrl"
        title="操作指引"
        type="info"
        :closable="false"
        class="guide-alert"
      >
        <p>1. 输入需要检测的新闻内容URL（例如：https://example.com/news/123）</p>
        <p>2. 点击"开始检测"按钮进行分析</p>
        <p>3. 等待系统返回检测结果</p>
      </el-alert>

      <el-input 
        v-model="inputUrl"
        placeholder="请输入需要检测的内容URL"
        class="url-input"
      >
        <template #append>
          <el-button 
            type="primary" 
            @click="startDetection"
            :loading="loading"
          >
            开始检测
          </el-button>
        </template>
      </el-input>

      <!-- 新增结果展示区域 -->
      <div v-if="results" class="result-container">
        <h3>检测结果</h3>
        <el-alert
          :title="results.isFake ? '疑似虚假内容' : '内容可信'"
          :type="results.isFake ? 'error' : 'success'"
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
              <label>分析结论：</label>
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

      <!-- 新增加载状态 -->
      <div v-if="loading" class="loading-container">
        <el-progress 
          :percentage="progress" 
          status="success" 
          :show-text="false"
        />
        <p>正在分析中，请稍候...</p>
      </div>
    </el-card>
  </div>
</template>

<script>
export default {
  name: 'HybridDetection',
  data() {
    return {
      inputUrl: '',
      loading: false,
      progress: 0,
      results: null
    }
  },
  methods: {
    async startDetection() {
      if (!this.inputUrl) {
        this.$message.warning('请输入检测URL')
        return
      }
      
      this.loading = true
      this.progress = 0
      const interval = setInterval(() => {
        this.progress = Math.min(this.progress + 10, 90)
      }, 500)

      try {
        // 模拟API调用
        const response = await this.$api.checkHybrid(this.inputUrl)
        this.results = {
          isFake: response.data.is_fake,
          confidence: response.data.confidence,
          description: response.data.description
        }
      } catch (error) {
        this.$message.error('检测失败：' + error.message)
      } finally {
        clearInterval(interval)
        this.loading = false
        this.progress = 100
      }
    },
    
    resetForm() {
      this.inputUrl = ''
      this.results = null
    }
  }
}
</script>

<style scoped>
/* 新增页面容器样式 */
.hybrid-detection {
  max-width: 1000px;
  margin: 0 auto;
  padding: 20px;
}

/* 新增页眉样式 */
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

/* 调整原卡片样式 */
.main-card {
  min-height: 400px;
  padding: 20px;
}

.guide-alert {
  margin: 20px 0;
  text-align: left;
}

.url-input {
  margin-top: 30px;
  max-width: 600px;
  margin-left: auto;
  margin-right: auto;  /* 新增居中效果 */
}

.result-container {
  margin-top: 30px;
  padding: 20px;
  background: #f5f7fa;
  border-radius: 4px;
}

.detail-item {
  background: white;
  padding: 15px;
  margin: 10px 0;
  border-radius: 4px;
}

.loading-container {
  margin-top: 30px;
  text-align: center;
}

.reset-btn {
  margin-top: 20px;
  width: 100%;
}
</style>
