<template>
  <div class="single-detection">
    <div class="page-header">
      <h1>单文本检测</h1>
      <p>在下方输入新闻文本，系统将分析其真实性</p>
    </div>

    <el-card>
      <el-form :model="form" ref="form" :rules="rules">
        <el-form-item prop="text">
          <el-input
            type="textarea"
            :rows="6"
            placeholder="请输入需要检测的新闻文本..."
            v-model="form.text"
          ></el-input>
        </el-form-item>

        <el-form-item>
          <div class="actions">
            <el-button type="primary" @click="submitForm" :loading="loading" :disabled="loading">
              开始检测
            </el-button>
            <el-button @click="resetForm" :disabled="loading">清空</el-button>
            <el-button type="text" @click="fillSampleTrue" :disabled="loading">填充真实新闻示例</el-button>
            <el-button type="text" @click="fillSampleFake" :disabled="loading">填充虚假新闻示例</el-button>
          </div>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- API状态卡片 -->
    <el-card class="api-status-card" v-if="apiError">
      <div slot="header" class="clearfix">
        <span>API连接状态</span>
        <el-button style="float: right; padding: 3px 0" type="text" @click="checkApiStatus">检查API状态</el-button>
      </div>
      <el-alert
        title="API连接错误"
        type="error"
        description="无法连接到后端API服务器，请确保API服务正在运行"
        show-icon
        :closable="false"
      >
      </el-alert>
      <p>错误详情: {{ apiErrorMsg }}</p>
      <p>请确保在端口5000启动了API服务器: <code>python api/app.py</code></p>
    </el-card>

    <!-- 原始响应调试卡片 -->
    <el-card class="debug-card" v-if="rawResponse">
      <div slot="header" class="clearfix">
        <span>API响应数据（调试信息）</span>
        <el-button style="float: right; padding: 3px 0" type="text" @click="rawResponse = null">关闭</el-button>
      </div>
      <pre>{{ JSON.stringify(rawResponse, null, 2) }}</pre>
    </el-card>

    <result-card 
      v-if="result" 
      :result="result" 
      @clear="result = null"
    />

    <loading-indicator 
      :loading="loading" 
      text="正在分析新闻文本，请稍候..."
    />
  </div>
</template>

<script>
import ResultCard from '@/components/ResultCard.vue'
import LoadingIndicator from '@/components/LoadingIndicator.vue'

export default {
  name: 'SingleDetection',
  components: {
    ResultCard,
    LoadingIndicator
  },
  data() {
    return {
      form: {
        text: ''
      },
      rules: {
        text: [
          { required: true, message: '请输入新闻文本', trigger: 'blur' },
          { min: 10, message: '文本至少需要10个字符', trigger: 'blur' }
        ]
      },
      loading: false,
      result: null,
      apiError: false,
      apiErrorMsg: '',
      rawResponse: null
    }
  },
  mounted() {
    // 组件加载时检查API状态
    this.checkApiStatus()
  },
  methods: {
    checkApiStatus() {
      this.$api.checkHealth()
        .then(response => {
          console.log('API状态检查成功:', response.data)
          this.apiError = false
          this.apiErrorMsg = ''
        })
        .catch(error => {
          console.error('API状态检查失败:', error)
          this.apiError = true
          this.apiErrorMsg = error.message || '未知错误'
        })
    },
    submitForm() {
      this.$refs.form.validate(valid => {
        if (valid) {
          this.detectNews()
        } else {
          return false
        }
      })
    },
    resetForm() {
      this.$refs.form.resetFields()
      this.result = null
      this.rawResponse = null
    },
    detectNews() {
      this.loading = true
      this.result = null
      this.rawResponse = null
      console.log('发送检测请求，文本长度:', this.form.text.length)
      
      this.$api.predictSingle(this.form.text)
        .then(response => {
          console.log('收到预测响应:', response)
          // 保存原始响应用于调试
          this.rawResponse = response.data
          
          if (response.data && response.data.success) {
            try {
              // 使用API服务中的格式化函数
              this.result = this.$api.formatSingleResponse(response.data)
              this.$message.success('检测完成')
            } catch (err) {
              console.error('处理响应数据出错:', err)
              this.$message.error('处理响应数据出错: ' + err.message)
            }
          } else {
            this.$message.error('检测失败: ' + (response.data.error || '未知错误'))
          }
        })
        .catch(error => {
          console.error('API请求失败:', error)
          this.apiError = true
          this.apiErrorMsg = error.message || '未知错误'
          this.$message.error('API请求失败，请检查服务是否正常运行')
        })
        .finally(() => {
          this.loading = false
        })
    },
    fillSampleTrue() {
      this.form.text = '北京冬奥会2022年2月4日开幕，中国代表团获得9金4银2铜的成绩。'
    },
    fillSampleFake() {
      this.form.text = '震惊！某明星深夜现身酒吧，与神秘人密会3小时'
    }
  }
}
</script>

<style scoped>
.single-detection {
  max-width: 1000px;
  margin: 0 auto;
}

.page-header {
  text-align: center;
  margin-bottom: 20px;
}

/* 已移除 detection-switch 相关样式 */
</style>