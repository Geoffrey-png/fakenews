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

// 示例图片数据（模拟从服务器获取）
const sampleImages = [
  '/image/1.jpg',
  '/image/2.jpg',
  '/image/3.jpg',
  '/image/4.jpg',
  '/image/5.jpg',
  '/image/6.jpg',
  '/image/7.jpg',
  '/image/8.jpg'
];

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
  
  // 生成假新闻解释
  generateExplanation(text, prediction) {
    return api.post('/generate_explanation', { text, prediction })
  },
  
  // 混合内容检测函数 (URL检测)
  checkHybrid(url) {
    // 目前暂时返回模拟数据
    // 真实实现应该调用后端API
    return new Promise((resolve) => {
      setTimeout(() => {
        // 随机决定是真实还是虚假新闻
        const isFake = Math.random() > 0.5;
        resolve({
          data: {
            is_fake: isFake,
            // 调整置信度范围：
            // 虚假新闻：85%-98% (更明显的虚假特征)
            // 真实新闻：80%-95% (较高的真实性)
            confidence: isFake ? Math.random() * 0.13 + 0.85 : Math.random() * 0.15 + 0.80,
            description: isFake ? 
              '系统检测到此内容包含多个可疑特征，多处细节不符合常理，可能为人工合成或刻意篡改的虚假内容。' :
              '系统未检测到明显造假特征，内容可信度较高，符合事实逻辑和常识判断。'
          }
        });
      }, 2000); // 模拟2秒延迟
    });
  },
  
  // 视频分析功能
  analyzeVideo(videoFile) {
    // 直接返回所有可用的图片
    return new Promise((resolve) => {
      // 模拟处理延迟
      setTimeout(() => {
        // 检查图片是否存在
        const checkImage = (path) => {
          return new Promise((resolveCheck) => {
            const img = new Image();
            img.onload = () => resolveCheck(true);
            img.onerror = () => resolveCheck(false);
            img.src = path;
          });
        };

        // 查找实际存在的图片（异步操作）
        Promise.all(sampleImages.map(path => checkImage(path)))
          .then(results => {
            // 过滤出存在的图片
            const availableImages = sampleImages.filter((_, index) => results[index]);
            
            // 如果没有找到任何图片，使用默认文本提示
            if (availableImages.length === 0) {
              resolve({
                data: {
                  success: false,
                  error: "未找到任何图片，请确保在frontend/public/image目录中放置了1.jpg、2.jpg等命名的图片",
                  frames: []
                }
              });
              return;
            }
            
            // 为每个帧添加检测结果
            const framesWithResults = availableImages.map((frame, index) => {
              return {
                frame_path: frame,
                frame_number: index + 1,
                is_fake: true,
                confidence: Math.random() * 0.13 + 0.85,
                manipulation_type: ['内容篡改', '面部替换', '背景修改', '对象插入'][index % 4]
              };
            });
            
            resolve({
              data: {
                success: true,
                frames: framesWithResults,
                video_name: videoFile ? videoFile.name : "样本视频",
                total_frames: framesWithResults.length,
                processing_time: 2.5, // 固定处理时间
                summary: `视频分析完成，找到${framesWithResults.length}个可疑帧。`
              }
            });
          });
      }, 1500); // 模拟1.5秒处理时间
    });
  },
  
  // 获取视频帧图片
  getVideoFrames() {
    // 实际应该从后端API获取，这里直接返回示例数据
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          data: {
            success: true,
            frames: sampleImages.map((path, index) => ({
              path,
              frame_number: index + 1
            }))
          }
        });
      }, 1000);
    });
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