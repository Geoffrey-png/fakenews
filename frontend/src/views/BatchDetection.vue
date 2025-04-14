<template>
  <div class="batch-detection">
    <div class="page-header">
      <h1>批量检测</h1>
      <p>在下方输入多条新闻文本，系统将批量分析其真实性</p>
    </div>

    <el-card>
      <el-form :model="form" ref="form" :rules="rules">
        <el-form-item label="输入方式">
          <el-radio-group v-model="inputMethod">
            <el-radio label="manual">手动输入</el-radio>
            <el-radio label="paste">批量粘贴</el-radio>
          </el-radio-group>
        </el-form-item>

        <el-form-item v-if="inputMethod === 'manual'" prop="texts">
          <div v-for="(item, index) in form.texts" :key="index" class="text-item">
            <div class="text-item-header">
              <h4>文本 #{{ index + 1 }}</h4>
              <el-button 
                v-if="form.texts.length > 1" 
                type="danger" 
                size="mini" 
                icon="el-icon-delete" 
                circle 
                @click="removeText(index)"
              ></el-button>
            </div>
            <el-input
              type="textarea"
              :rows="3"
              :placeholder="`请输入第 ${index + 1} 条需要检测的新闻文本...`"
              v-model="form.texts[index]"
            ></el-input>
          </div>
          <div class="add-text-btn">
            <el-button type="primary" plain icon="el-icon-plus" @click="addText">添加更多文本</el-button>
          </div>
        </el-form-item>

        <el-form-item v-else prop="bulkText">
          <p class="tips">请输入多条新闻文本，每行一条</p>
          <el-input
            type="textarea"
            :rows="10"
            placeholder="请粘贴需要检测的新闻文本，每行一条..."
            v-model="form.bulkText"
          ></el-input>
        </el-form-item>

        <el-form-item>
          <div class="actions">
            <el-button type="primary" @click="submitForm" :loading="loading" :disabled="loading">
              开始批量检测
            </el-button>
            <el-button @click="resetForm" :disabled="loading">清空</el-button>
            <el-button type="text" @click="fillSample" :disabled="loading">填充示例数据</el-button>
          </div>
        </el-form-item>
      </el-form>
    </el-card>

    <div v-if="results.length > 0" class="results-container">
      <el-card>
        <div slot="header" class="clearfix">
          <span>批量检测结果 ({{ results.length }}条)</span>
          <div style="float: right">
            <el-button type="text" @click="sortResults('index')">原始顺序</el-button>
            <el-button type="text" @click="sortResults('confidence')">按置信度排序</el-button>
            <el-button style="padding: 3px 0; margin-left: 10px;" type="text" @click="results = []">清除结果</el-button>
          </div>
        </div>
        
        <div class="stats-summary">
          <div class="stats-item">
            <div class="stats-label">总计检测</div>
            <div class="stats-value">{{ results.length }}</div>
          </div>
          <div class="stats-item">
            <div class="stats-label">真实新闻</div>
            <div class="stats-value true-news">{{ getTrueNewsCount() }}</div>
          </div>
          <div class="stats-item">
            <div class="stats-label">虚假新闻</div>
            <div class="stats-value fake-news">{{ getFakeNewsCount() }}</div>
          </div>
          <div class="stats-item">
            <div class="stats-label">处理时间</div>
            <div class="stats-value">{{ getTotalTime() }}秒</div>
          </div>
        </div>

        <el-table
          :data="results"
          style="width: 100%"
          stripe
          border
          v-loading="tableLoading"
        >
          <el-table-column
            label="#"
            width="60"
            align="center"
          >
            <template slot-scope="scope">
              {{ scope.row.index + 1 }}
            </template>
          </el-table-column>
          <el-table-column
            prop="text"
            label="新闻文本"
            show-overflow-tooltip
          >
            <template slot-scope="scope">
              <div class="news-text-cell">{{ scope.row.text }}</div>
            </template>
          </el-table-column>
          <el-table-column
            prop="prediction.label"
            label="检测结果"
            width="120"
            align="center"
          >
            <template slot-scope="scope">
              <el-tag 
                :type="scope.row.prediction.label === '真实新闻' ? 'success' : 'danger'"
                effect="dark"
              >
                {{ scope.row.prediction.label }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column
            prop="prediction.confidence"
            label="置信度"
            width="120"
            align="center"
          >
            <template slot-scope="scope">
              <el-progress 
                type="circle" 
                :percentage="getPercentage(scope.row.prediction.confidence)" 
                :color="getConfidenceColor(scope.row.prediction.confidence)"
                :width="40"
              ></el-progress>
            </template>
          </el-table-column>
          <el-table-column
            label="操作"
            width="120"
            align="center"
          >
            <template slot-scope="scope">
              <el-button 
                size="mini" 
                type="primary" 
                @click="showDetail(scope.row)"
                icon="el-icon-view" 
                circle
              ></el-button>
              <el-button 
                size="mini" 
                type="danger" 
                @click="removeResult(scope.$index)" 
                icon="el-icon-delete" 
                circle
              ></el-button>
            </template>
          </el-table-column>
          
          <!-- 假新闻解释简介列 -->
          <el-table-column
            label="假新闻理由"
            min-width="200"
            show-overflow-tooltip
            v-if="hasFakeNewsOrForExplanation"
          >
            <template slot-scope="scope">
              <div v-if="scope.row.prediction.label === '虚假新闻' && scope.row.explanation" 
                   class="short-explanation"
                   @click="showDetail(scope.row)">
                <i class="el-icon-warning-outline"></i>
                <span>{{ getShortExplanation(scope.row.explanation) }}</span>
              </div>
              <div v-else-if="scope.row.prediction.label === '虚假新闻' && !scope.row.explanation"
                   class="get-explanation-button">
                <el-button 
                  type="text" 
                  size="small"
                  @click="generateExplanation(scope.row)"
                  :loading="scope.row.loadingExplanation"
                >
                  <i class="el-icon-question"></i> 
                  获取解释
                </el-button>
              </div>
              <span v-else>-</span>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </div>

    <!-- 详情对话框 -->
    <el-dialog
      title="检测详情"
      :visible.sync="detailDialogVisible"
      width="500px"
    >
      <div v-if="currentDetail" class="detail-dialog">
        <div class="detail-text">
          <h4>新闻文本</h4>
          <p>{{ currentDetail.text }}</p>
        </div>
        <div class="detail-result">
          <h4>检测结果</h4>
          <el-tag 
            :type="currentDetail.prediction.label === '真实新闻' ? 'success' : 'danger'"
            effect="dark"
            size="medium"
          >
            {{ currentDetail.prediction.label }}
          </el-tag>
          <span class="confidence">(置信度: {{ formatPercentage(currentDetail.prediction.confidence) }}%)</span>
        </div>
        <div class="detail-probs">
          <h4>详细分析</h4>
          <div class="prob-item">
            <div class="prob-label">真实新闻概率:</div>
            <el-progress 
              :percentage="getPercentage(currentDetail.prediction.probabilities['真实新闻'])" 
              :format="p => p + '%'"
              :color="getConfidenceColor(currentDetail.prediction.probabilities['真实新闻'])"
            ></el-progress>
          </div>
          <div class="prob-item">
            <div class="prob-label">虚假新闻概率:</div>
            <el-progress 
              :percentage="getPercentage(currentDetail.prediction.probabilities['虚假新闻'])" 
              :format="p => p + '%'"
              :color="getConfidenceColor(currentDetail.prediction.probabilities['虚假新闻'])"
            ></el-progress>
          </div>
        </div>
        
        <!-- 添加解释部分 -->
        <div v-if="currentDetail.prediction.label === '虚假新闻'" class="detail-explanation">
          <h4>为什么这是假新闻?</h4>
          
          <div v-if="!currentDetail.explanation && !currentDetail.loadingExplanation" class="explanation-request">
            <el-button 
              type="primary" 
              size="medium" 
              @click="generateExplanation(currentDetail)" 
              :loading="currentDetail.loadingExplanation"
              icon="el-icon-question"
            >
              获取大模型解释
            </el-button>
            <span class="explanation-hint">点击按钮了解为何判定为假新闻</span>
          </div>
          
          <div v-if="currentDetail.loadingExplanation" class="explanation-loading">
            <i class="el-icon-loading"></i> 
            <span>正在生成解释，请稍候...</span>
          </div>
          
          <div v-if="currentDetail.explanation" class="explanation-box">
            <i class="el-icon-warning-outline"></i>
            <div v-html="formatExplanationText(currentDetail.explanation)"></div>
          </div>
        </div>
      </div>
      <span slot="footer" class="dialog-footer">
        <el-button @click="detailDialogVisible = false">关闭</el-button>
      </span>
    </el-dialog>

    <loading-indicator 
      :loading="loading" 
      text="正在批量分析新闻文本，请稍候..."
    />
  </div>
</template>

<script>
import LoadingIndicator from '@/components/LoadingIndicator.vue'

export default {
  name: 'BatchDetection',
  components: {
    LoadingIndicator
  },
  data() {
    return {
      inputMethod: 'manual',
      form: {
        texts: [''],
        bulkText: ''
      },
      rules: {
        bulkText: [
          { required: true, message: '请输入新闻文本', trigger: 'blur' }
        ]
      },
      loading: false,
      tableLoading: false,
      results: [],
      detailDialogVisible: false,
      currentDetail: null
    }
  },
  computed: {
    // 计算是否有包含解释的假新闻或需要显示解释按钮
    hasFakeNewsOrForExplanation() {
      return this.results.some(item => 
        item.prediction && 
        item.prediction.label === '虚假新闻'
      );
    }
  },
  methods: {
    addText() {
      this.form.texts.push('')
    },
    removeText(index) {
      this.form.texts.splice(index, 1)
    },
    submitForm() {
      if (this.inputMethod === 'manual') {
        // 检查至少有一条非空文本
        const validTexts = this.form.texts.filter(text => text.trim().length > 0)
        if (validTexts.length === 0) {
          this.$message.warning('请至少输入一条有效的新闻文本')
          return
        }
        
        this.batchDetect(validTexts)
      } else {
        // 检查批量文本是否为空
        if (!this.form.bulkText.trim()) {
          this.$message.warning('请输入需要检测的新闻文本')
          return
        }
        
        // 将批量文本拆分为数组
        const texts = this.form.bulkText.split('\n')
          .map(text => text.trim())
          .filter(text => text.length > 0)
        
        if (texts.length === 0) {
          this.$message.warning('未找到有效的新闻文本')
          return
        }
        
        this.batchDetect(texts)
      }
    },
    batchDetect(texts) {
      this.loading = true
      this.tableLoading = true
      
      this.$api.predictBatch(texts)
        .then(response => {
          if (response.data.success) {
            this.processResults(texts, response.data.results)
          } else {
            this.$message.error('批量检测失败: ' + response.data.error)
          }
        })
        .catch(error => {
          console.error('API请求失败:', error)
          this.$message.error('API请求失败，请检查服务是否正常运行')
        })
        .finally(() => {
          this.loading = false
          this.tableLoading = false
        })
    },
    processResults(texts, apiResults) {
      this.results = texts.map((text, index) => {
        // 获取对应的API结果
        const result = apiResults[index];
        
        // 确保有一个有效的process_time值
        const processTime = this.ensureNumber(result.process_time || result.processing_time, 0.1);
        
        // 确保有一个有效的prediction对象
        let prediction = result.prediction || {};
        
        // 如果prediction中有confidence对象，使用它的值
        if (prediction.confidence && typeof prediction.confidence === 'object') {
          // 获取标签
          const label = prediction.label || (prediction.label_id === 0 ? '真实新闻' : '虚假新闻');
          const isRealNews = label === '真实新闻';
          
          // 获取正确的confidence值
          const trueNewsConf = this.ensureNumber(prediction.confidence['真实新闻'], 0.5);
          const fakeNewsConf = this.ensureNumber(prediction.confidence['虚假新闻'], 0.5);
          
          // 设置confidence为当前标签对应的置信度
          prediction.confidence = isRealNews ? trueNewsConf : fakeNewsConf;
          
          // 确保probabilities存在
          prediction.probabilities = {
            '真实新闻': trueNewsConf,
            '虚假新闻': fakeNewsConf
          };
        } else if (!prediction.confidence) {
          // 如果没有confidence，设置默认值
          prediction.confidence = 0.5;
          prediction.probabilities = {
            '真实新闻': 0.5,
            '虚假新闻': 0.5
          };
        }
        
        // 返回格式化后的结果对象
        return {
          index,
          text,
          prediction,
          process_time: processTime,
          loadingExplanation: false
        };
      });
      
      // 打印处理后的结果，用于调试
      console.log('处理后的批量检测结果:', this.results);
    },
    ensureNumber(value, defaultValue = 0) {
      if (value === undefined || value === null) return defaultValue;
      const num = parseFloat(value);
      return isNaN(num) ? defaultValue : num;
    },
    resetForm() {
      if (this.inputMethod === 'manual') {
        this.form.texts = ['']
      } else {
        this.form.bulkText = ''
      }
      this.results = []
    },
    fillSample() {
      const sampleTexts = [
        '央视网消息（新闻联播）：4月13日，西北、华北、黄淮等部分地区仍有大风，东北等地还有雨雪天气。中央气象台4月13日继续发布大风黄色、暴雪蓝色、沙尘暴蓝色预警。',
        '震惊！某明星深夜现身酒吧，与神秘人密会3小时',
        '中国科学家在量子计算领域取得重大突破，实现了72量子比特的纠缠态。',
        '惊爆！研究表明喝水会导致癌症，专家建议立即停止饮用！',
        '近日，多家零售商超、电商平台等企业密集发声，将多举措助力中国优质外贸企业开拓国内市场。据不完全统计，截至4月13日，已有京东、盒马、永辉超市、华润万家、物美等近20家企业表示，将发挥各自优势助力外贸企业拓宽渠道。'
      ]
      
      if (this.inputMethod === 'manual') {
        this.form.texts = [...sampleTexts]
      } else {
        this.form.bulkText = sampleTexts.join('\n')
      }
    },
    showDetail(row) {
      this.currentDetail = row
      this.detailDialogVisible = true
    },
    removeResult(index) {
      this.results.splice(index, 1)
    },
    getConfidenceColor(confidence) {
      // 确保值是有效数字
      const num = this.ensureNumber(confidence, 0);
      if (num < 0.6) return '#F56C6C'
      if (num < 0.8) return '#E6A23C'
      return '#67C23A'
    },
    getTrueNewsCount() {
      return this.results.filter(item => item.prediction.label === '真实新闻').length
    },
    getFakeNewsCount() {
      return this.results.filter(item => item.prediction.label === '虚假新闻').length
    },
    getTotalTime() {
      if (this.results.length === 0) return 0;
      const totalTime = this.results.reduce((sum, item) => {
        // 确保process_time是有效数字
        const time = this.ensureNumber(item.process_time, 0);
        return sum + time;
      }, 0);
      return totalTime.toFixed(2);
    },
    sortResults(type) {
      if (type === 'index') {
        this.results.sort((a, b) => a.index - b.index);
      } else if (type === 'confidence') {
        this.results.sort((a, b) => {
          // 确保confidence是有效数字
          const confA = this.ensureNumber(b.prediction.confidence, 0);
          const confB = this.ensureNumber(a.prediction.confidence, 0);
          return confA - confB;
        });
      }
    },
    getPercentage(value) {
      // 确保值是有效数字
      const num = this.ensureNumber(value, 0.5);
      // 转换为百分比并返回整数
      return parseFloat((num * 100).toFixed(0));
    },
    formatPercentage(value) {
      // 确保值是有效数字
      const num = this.ensureNumber(value, 0.5);
      // 转换为百分比并返回两位小数
      return (num * 100).toFixed(2);
    },
    formatExplanationText(text) {
      // 将文本中的换行符转换为HTML换行
      return text.replace(/\n/g, '<br>');
    },
    getShortExplanation(explanation) {
      if (!explanation) return '';
      
      // 获取第一行作为简短解释
      const firstLine = explanation.split('\n')[0];
      
      // 如果第一行太长，则截取适当长度
      if (firstLine.length > 50) {
        return firstLine.substring(0, 50) + '...';
      }
      
      return firstLine;
    },
    generateExplanation(item) {
      if (!item || !item.text) {
        this.$message.error('无法获取解释: 缺少新闻文本');
        return;
      }
      
      // 设置加载状态
      this.$set(item, 'loadingExplanation', true);
      
      // 调用API获取解释
      this.$api.generateExplanation(item.text, item.prediction)
        .then(response => {
          console.log('解释生成响应:', response.data);
          
          if (response.data && response.data.success) {
            this.$set(item, 'explanation', response.data.explanation);
            this.$message.success('解释生成成功');
          } else {
            this.$message.error('解释生成失败: ' + (response.data.message || '未知错误'));
          }
        })
        .catch(error => {
          console.error('解释API请求失败:', error);
          this.$message.error('解释生成失败，请稍后再试');
        })
        .finally(() => {
          this.$set(item, 'loadingExplanation', false);
        });
    }
  }
}
</script>

<style scoped>
.batch-detection {
  max-width: 1200px;
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

.text-item {
  margin-bottom: 15px;
  border: 1px solid #EBEEF5;
  border-radius: 4px;
  padding: 10px;
  background-color: #FAFAFA;
}

.text-item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.text-item-header h4 {
  margin: 0;
  color: #606266;
}

.add-text-btn {
  margin-top: 10px;
  text-align: center;
}

.tips {
  font-size: 14px;
  color: #909399;
  margin-bottom: 10px;
}

.actions {
  display: flex;
  justify-content: start;
  flex-wrap: wrap;
  gap: 10px;
}

.results-container {
  margin-top: 30px;
}

.stats-summary {
  display: flex;
  justify-content: space-around;
  margin-bottom: 20px;
  background-color: #f7f9fc;
  padding: 15px;
  border-radius: 4px;
  flex-wrap: wrap;
}

.stats-item {
  text-align: center;
  padding: 0 15px;
}

.stats-label {
  font-size: 14px;
  color: #606266;
  margin-bottom: 5px;
}

.stats-value {
  font-size: 24px;
  font-weight: bold;
  color: #409EFF;
}

.stats-value.true-news {
  color: #67C23A;
}

.stats-value.fake-news {
  color: #F56C6C;
}

.news-text-cell {
  max-width: 400px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.detail-dialog {
  max-height: 60vh;
  overflow-y: auto;
}

.detail-text {
  margin-bottom: 20px;
}

.detail-text h4, .detail-result h4, .detail-probs h4 {
  margin-top: 0;
  margin-bottom: 10px;
  color: #606266;
  font-size: 16px;
}

.detail-text p {
  padding: 10px;
  background-color: #f7f7f7;
  border-radius: 4px;
  margin: 0;
  white-space: pre-wrap;
}

.detail-result {
  margin-bottom: 20px;
}

.confidence {
  margin-left: 10px;
  color: #909399;
}

.prob-item {
  margin-bottom: 15px;
}

.prob-label {
  margin-bottom: 5px;
  font-weight: 600;
}

.detail-explanation {
  margin-top: 20px;
  border-top: 1px solid #EBEEF5;
  padding-top: 15px;
}

.detail-explanation h4 {
  color: #F56C6C;
  margin-bottom: 10px;
}

.explanation-box {
  background-color: #FEF0F0;
  border-radius: 4px;
  padding: 15px;
  display: flex;
  align-items: flex-start;
}

.explanation-box i {
  color: #F56C6C;
  font-size: 18px;
  margin-right: 10px;
  margin-top: 3px;
}

.explanation-box div {
  flex: 1;
  line-height: 1.6;
}

.short-explanation {
  display: flex;
  align-items: center;
  color: #F56C6C;
  cursor: pointer;
  padding: 5px;
  border-radius: 4px;
  transition: background-color 0.3s;
}

.short-explanation:hover {
  background-color: #FEF0F0;
}

.short-explanation i {
  margin-right: 5px;
  font-size: 16px;
}

.explanation-request {
  margin-bottom: 10px;
}

.explanation-hint {
  margin-left: 10px;
  color: #909399;
}

.explanation-loading {
  margin-bottom: 10px;
}
</style> 