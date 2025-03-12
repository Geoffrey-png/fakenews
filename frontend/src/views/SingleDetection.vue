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

    <!-- 删除原始响应调试卡片 -->

    <result-card 
      v-if="result" 
      :result="result" 
      @clear="result = null"
    >
      <template v-slot:extra-content>
        <!-- 添加解释按钮和解释显示 -->
        <div v-if="result.prediction.label === '虚假新闻'" class="fake-news-explanation">
          <div v-if="!explanation && !loadingExplanation" class="explanation-request">
            <el-button 
              type="primary" 
              size="medium" 
              @click="getExplanation" 
              :loading="loadingExplanation"
              icon="el-icon-question"
            >
              获取大模型解释
            </el-button>
            <span class="explanation-hint">点击按钮了解为何判定为假新闻</span>
          </div>
          
          <div v-if="loadingExplanation" class="explanation-loading">
            <i class="el-icon-loading"></i> 
            <span>正在生成解释，请稍候...</span>
          </div>
          
          <div v-if="explanation" class="explanation-content-container">
            <h3>为什么系统认为这是假新闻?</h3>
            <div class="explanation-content">
              <i class="el-icon-warning-outline"></i>
              <div v-html="formatExplanation(explanation)"></div>
            </div>
          </div>
        </div>
      </template>
    </result-card>

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
      loadingExplanation: false,
      explanation: null,
      result: null,
      apiError: false,
      apiErrorMsg: '',
      rawResponse: null // 删除这一行
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
      this.explanation = null
      this.rawResponse = null // 删除这一行
    },
    detectNews() {
      this.loading = true
      this.result = null
      this.explanation = null
      this.rawResponse = null // 删除这一行
      console.log('发送检测请求，文本长度:', this.form.text.length)
      
      this.$api.predictSingle(this.form.text)
        .then(response => {
          console.log('收到预测响应:', response)
          // 保存原始响应用于调试
          this.rawResponse = response.data // 删除这一行
          
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
    getExplanation() {
      if (!this.result || !this.form.text) {
        this.$message.error('无法获取解释: 缺少新闻文本或检测结果')
        return
      }
      
      this.loadingExplanation = true
      
      // 构造请求数据
      const requestData = {
        text: this.form.text,
        prediction: {
          label: this.result.prediction.label,
          confidence: this.result.prediction.confidence
        }
      }
      
      // 调用API获取解释
      this.$api.generateExplanation(this.form.text, this.result.prediction)
        .then(response => {
          console.log('解释生成响应:', response.data)
          
          if (response.data && response.data.success) {
            this.explanation = response.data.explanation
            this.$message.success('解释生成成功')
          } else {
            this.$message.error('解释生成失败: ' + (response.data.message || '未知错误'))
          }
        })
        .catch(error => {
          console.error('解释API请求失败:', error)
          this.$message.error('解释生成失败，请稍后再试')
        })
        .finally(() => {
          this.loadingExplanation = false
        })
    },
    fillSampleTrue() {
      this.form.text = '北京冬奥会2022年2月4日开幕，中国代表团获得9金4银2铜的成绩。'
    },
    fillSampleFake() {
      this.form.text = '震惊！某明星深夜现身酒吧，与神秘人密会3小时'
    },
    formatExplanation(explanation) {
      // 实现格式化解释的逻辑
      return explanation.replace(/\n/g, '<br>')
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

.page-header h1 {
  color: #409EFF;
  margin-bottom: 10px;
}

.page-header p {
  color: #606266;
}

.actions {
  display: flex;
  justify-content: start;
  flex-wrap: wrap;
  gap: 10px;
}

.api-status-card {
  margin-top: 20px;
  margin-bottom: 20px;
  background-color: #FEF0F0;
}

.debug-card {
  margin-top: 20px;
  margin-bottom: 20px;
  background-color: #F0F9EB;
  overflow: auto;
}

.debug-card pre {
  white-space: pre-wrap;
  font-family: monospace;
  font-size: 12px;
}

.fake-news-explanation {
  margin-top: 20px;
  padding: 16px;
  background-color: #FFF8F8;
  border-radius: 4px;
  border-left: 4px solid #F56C6C;
}

.explanation-request {
  display: flex;
  align-items: center;
  margin-bottom: 10px;
}

.explanation-hint {
  margin-left: 10px;
  color: #606266;
  font-size: 14px;
}

.explanation-loading {
  display: flex;
  align-items: center;
  margin: 20px 0;
  color: #409EFF;
}

.explanation-loading i {
  margin-right: 10px;
}

.explanation-content-container h3 {
  margin-bottom: 16px;
  color: #F56C6C;
}

.explanation-content {
  display: flex;
  align-items: flex-start;
  background-color: #FFF;
  padding: 16px;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.explanation-content i {
  margin-right: 10px;
  color: #F56C6C;
  font-size: 20px;
}

.explanation-content div {
  flex: 1;
  line-height: 1.6;
}
</style> 