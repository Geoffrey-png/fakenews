import axios from 'axios'

// 创建axios实例
const api = axios.create({
  // 明确指定baseURL，确保连接到正确的API服务器
  baseURL: 'http://10.101.64.214:5000',
  timeout: 30000 // 请求超时时间
})

// 请求拦截器
api.interceptors.request.use(
  config => {
    // 在发送请求之前可以做一些处理
    console.log('发送请求到:', config.url, config.data)
    return config
  },
  error => {
    // 处理请求错误
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  response => {
    // 对响应数据做处理
    console.log('接收响应:', response.config.url, response.data)
    return response
  },
  error => {
    // 处理响应错误
    console.error('响应错误:', error)
    return Promise.reject(error)
  }
)

// 工具函数：确保值是有效的数字
function ensureNumber(value, defaultValue = 0) {
  const num = parseFloat(value);
  return isNaN(num) ? defaultValue : num;
}

// API服务
const apiService = {
  // 健康检查
  checkHealth() {
    return api.get('/health')
  },

  // 单文本预测
  predictSingle(text) {
    return api.post('/predict', { text })
  },

  // 批量预测
  predictBatch(texts) {
    return api.post('/batch_predict', { texts })
  },
  
  // 转换单文本响应为标准格式
  formatSingleResponse(data) {
    console.log('格式化前的原始数据:', JSON.stringify(data));
    
    // 确保process_time字段存在
    if (data.process_time === undefined && data.processing_time !== undefined) {
      data.process_time = ensureNumber(data.processing_time, 0.1);
    } else if (data.process_time === undefined) {
      data.process_time = 0.1; // 默认处理时间
    }
    
    // 确保有一个可用的prediction对象
    if (!data.prediction) {
      // 获取class和confidence，确保它们是有效数字
      const classValue = data.class !== undefined ? data.class : 1; // 默认为1（虚假新闻）
      const isRealNews = classValue === 0;
      const confidenceValue = ensureNumber(data.confidence, 0.5); // 使用0.5作为默认值
      
      data.prediction = {
        label: isRealNews ? '真实新闻' : '虚假新闻',
        label_id: classValue,
        confidence: confidenceValue
      }
      
      // 为probabilities设置默认值
      data.prediction.probabilities = {
        '真实新闻': isRealNews ? confidenceValue : 1 - confidenceValue,
        '虚假新闻': isRealNews ? 1 - confidenceValue : confidenceValue
      }
    }
    
    // 处理confidence对象 - 这是用户指定的用于概率显示的对象
    if (data.prediction.confidence && typeof data.prediction.confidence === 'object') {
      // 如果confidence是对象，则直接使用它的值作为probabilities
      data.prediction.probabilities = {
        '真实新闻': ensureNumber(data.prediction.confidence['真实新闻'], 0.5),
        '虚假新闻': ensureNumber(data.prediction.confidence['虚假新闻'], 0.5)
      }
      
      // 同时保留一个标量形式的confidence作为总体置信度
      const label = data.prediction.label;
      if (label === '真实新闻') {
        data.prediction.confidenceValue = data.prediction.probabilities['真实新闻'];
      } else {
        data.prediction.confidenceValue = data.prediction.probabilities['虚假新闻'];
      }
    } else if (typeof data.prediction.confidence === 'number') {
      // 如果confidence是数字，则使用它设置probabilities
      const isRealNews = data.prediction.label === '真实新闻' || data.prediction.label_id === 0;
      const confidenceValue = ensureNumber(data.prediction.confidence, 0.5);
      
      data.prediction.probabilities = {
        '真实新闻': isRealNews ? confidenceValue : 1 - confidenceValue,
        '虚假新闻': isRealNews ? 1 - confidenceValue : confidenceValue
      }
      
      data.prediction.confidenceValue = confidenceValue;
    } else {
      // 确保有默认值
      data.prediction.probabilities = data.prediction.probabilities || {
        '真实新闻': 0.5,
        '虚假新闻': 0.5
      };
      
      // 确保probabilities中的值是有效数字
      data.prediction.probabilities['真实新闻'] = ensureNumber(data.prediction.probabilities['真实新闻'], 0.5);
      data.prediction.probabilities['虚假新闻'] = ensureNumber(data.prediction.probabilities['虚假新闻'], 0.5);
      
      // 设置默认的confidenceValue
      const label = data.prediction.label;
      if (label === '真实新闻') {
        data.prediction.confidenceValue = data.prediction.probabilities['真实新闻'];
      } else {
        data.prediction.confidenceValue = data.prediction.probabilities['虚假新闻'];
      }
    }
    
    console.log('格式化后的数据:', JSON.stringify(data));
    return data
  }
}

export default apiService 