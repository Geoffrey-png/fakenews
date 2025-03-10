<template>
  <el-card class="result-card" :body-style="{ padding: '0' }">
    <div slot="header" class="clearfix">
      <span>{{ title }}</span>
      <el-button style="float: right; padding: 3px 0" type="text" @click="$emit('clear')">
        {{ clearButtonText }}
      </el-button>
    </div>
    
    <div class="result-content">
      <el-result 
        :icon="result.prediction.label === '真实新闻' ? 'success' : 'error'"
        :title="result.prediction.label"
        :subTitle="'置信度: ' + formatPercentage(getConfidenceValue()) + '%'"
      >
        <template slot="extra">
          <div class="detail-info">
            <h3>详细信息</h3>
            <div class="probabilities">
              <div class="prob-item">
                <span>真实新闻概率:</span>
                <el-progress 
                  :percentage="getPercentage(result.prediction.probabilities['真实新闻'])" 
                  :color="customColorMethod(result.prediction.probabilities['真实新闻'])"
                ></el-progress>
              </div>
              <div class="prob-item">
                <span>虚假新闻概率:</span>
                <el-progress 
                  :percentage="getPercentage(result.prediction.probabilities['虚假新闻'])" 
                  :color="customColorMethod(result.prediction.probabilities['虚假新闻'])"
                ></el-progress>
              </div>
            </div>
            <p class="proc-time">处理时间: {{ result.process_time || 0 }}秒</p>
            
            <slot name="extra-content"></slot>
          </div>
        </template>
      </el-result>
    </div>
  </el-card>
</template>

<script>
export default {
  name: 'ResultCard',
  props: {
    result: {
      type: Object,
      required: true
    },
    title: {
      type: String,
      default: '检测结果'
    },
    clearButtonText: {
      type: String,
      default: '清除结果'
    }
  },
  methods: {
    // 获取置信度值
    getConfidenceValue() {
      // 如果有confidenceValue属性，使用它
      if (this.result.prediction.confidenceValue !== undefined) {
        return this.result.prediction.confidenceValue;
      }
      // 否则，根据标签取对应的概率值
      const label = this.result.prediction.label;
      if (label === '真实新闻') {
        return this.result.prediction.probabilities['真实新闻'];
      } else {
        return this.result.prediction.probabilities['虚假新闻'];
      }
    },
    // 确保值是有效的数字
    ensureNumber(value, defaultValue = 0) {
      const num = parseFloat(value);
      return isNaN(num) ? defaultValue : num;
    },
    // 将概率值转换为百分比
    getPercentage(value) {
      const num = this.ensureNumber(value, 0.5);
      return parseFloat((num * 100).toFixed(1));
    },
    // 格式化百分比显示
    formatPercentage(value) {
      return (this.ensureNumber(value, 0) * 100).toFixed(2);
    },
    // 根据概率值返回不同的颜色
    customColorMethod(percentage) {
      const num = this.ensureNumber(percentage, 0);
      if (num < 0.3) return '#67C23A'
      if (num < 0.7) return '#E6A23C'
      return '#F56C6C'
    }
  }
}
</script>

<style scoped>
.result-card {
  margin-top: 20px;
  background-color: #f7f9fc;
}

.result-content {
  text-align: center;
}

.detail-info {
  margin-top: 20px;
  text-align: left;
  max-width: 500px;
  margin-left: auto;
  margin-right: auto;
}

.probabilities {
  margin-top: 20px;
}

.prob-item {
  margin-bottom: 15px;
}

.prob-item span {
  display: block;
  margin-bottom: 5px;
  font-weight: 600;
}

.proc-time {
  text-align: right;
  color: #909399;
  font-size: 0.8rem;
  margin-top: 20px;
}
</style> 