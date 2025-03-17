# 中文假新闻检测API服务

本API服务提供了一个简单的REST接口，用于检测中文新闻文本是真实新闻还是虚假新闻（谣言）。

## 功能特点

- 基于BERT深度学习模型进行新闻真伪检测
- 提供单文本和批量文本预测接口
- 返回详细的置信度信息
- 支持跨域请求
- 提供健康检查端点
- 完善的错误处理和日志记录

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

### 开发环境

```bash
# 直接启动Flask开发服务器
cd api
python app.py
```

### 生产环境

```bash
# 使用gunicorn启动
cd api
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## API端点

### 健康检查

```
GET /health
```

**响应示例**:

```json
{
  "status": "ok",
  "timestamp": 1647832846.1234567,
  "model_loaded": true
}
```

### 单文本预测

```
POST /predict
```

**请求体**:

```json
{
  "text": "需要检测的新闻文本"
}
```

**响应示例**:

```json
{
  "success": true,
  "prediction": {
    "label": "真实新闻",
    "label_id": 0,
    "confidence": {
      "真实新闻": 0.9876,
      "虚假新闻": 0.0124
    }
  },
  "processing_time": 0.1234
}
```

### 批量预测

```
POST /batch_predict
```

**请求体**:

```json
{
  "texts": ["新闻文本1", "新闻文本2", "..."]
}
```

**响应示例**:

```json
{
  "success": true,
  "results": [
    {
      "success": true,
      "prediction": {
        "label": "真实新闻",
        "label_id": 0,
        "confidence": {
          "真实新闻": 0.9876,
          "虚假新闻": 0.0124
        }
      }
    },
    {
      "success": true,
      "prediction": {
        "label": "虚假新闻",
        "label_id": 1,
        "confidence": {
          "真实新闻": 0.0321,
          "虚假新闻": 0.9679
        }
      }
    }
  ],
  "processing_time": 0.2345
}
```

## 错误处理

服务会返回适当的HTTP状态码和错误信息：

```json
{
  "success": false,
  "error": "错误描述"
}
```

## 环境变量

服务可以通过以下环境变量进行配置：

- `API_HOST`: 服务监听的主机地址（默认: `0.0.0.0`）
- `API_PORT`: 服务监听的端口（默认: `5000`）
- `DEBUG`: 是否启用调试模式（默认: `false`）
- `CORS_ORIGINS`: 允许的跨域源，以逗号分隔（默认: `*`）
- `MAX_SEQUENCE_LENGTH`: 文本序列的最大长度（默认: `128`）
- `BATCH_SIZE`: 批处理大小（默认: `8`）

## 前端集成示例

### 使用Fetch API

```javascript
// 单文本预测
async function predictNews(newsText) {
  try {
    const response = await fetch('http://your-api-host:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: newsText }),
    });
    
    const result = await response.json();
    if (result.success) {
      return result.prediction;
    } else {
      throw new Error(result.error || '预测失败');
    }
  } catch (error) {
    console.error('预测出错:', error);
    throw error;
  }
}

// 使用示例
const newsText = document.getElementById('news-input').value;
predictNews(newsText)
  .then(prediction => {
    document.getElementById('result-label').textContent = prediction.label;
    document.getElementById('result-confidence').textContent = 
      `置信度: ${(prediction.confidence[prediction.label] * 100).toFixed(2)}%`;
  })
  .catch(error => {
    document.getElementById('error-message').textContent = error.message;
  });
```

### 使用Axios

```javascript
// 批量预测
async function batchPredictNews(newsTexts) {
  try {
    const response = await axios.post('http://your-api-host:5000/batch_predict', {
      texts: newsTexts
    });
    
    if (response.data.success) {
      return response.data.results;
    } else {
      throw new Error(response.data.error || '批量预测失败');
    }
  } catch (error) {
    console.error('批量预测出错:', error);
    throw error;
  }
}
```

## 测试API

服务附带了一个测试脚本，可以验证API是否正常工作：

```bash
cd api
python test_api.py
``` 