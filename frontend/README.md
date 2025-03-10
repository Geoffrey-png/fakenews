# 假新闻检测系统前端

这是假新闻检测系统的前端部分，使用Vue.js和Element UI构建。

## 技术栈

- Vue.js 2.x
- Element UI
- Axios
- Vue Router

## 项目结构

```
frontend/
├── public/                      # 静态资源
│   ├── index.html               # HTML入口文件
│   └── img/                     # 图片资源目录
│
├── src/                         # 源代码
│   ├── assets/                  # 静态资源
│   │   └── styles.css           # 全局样式文件
│   │
│   ├── components/              # 公共组件
│   │   ├── LoadingIndicator.vue # 加载状态指示器组件
│   │   └── ResultCard.vue       # 检测结果卡片组件
│   │
│   ├── views/                   # 页面组件
│   │   ├── Home.vue             # 首页 - 展示系统概览和功能入口
│   │   ├── SingleDetection.vue  # 单文本检测页面 - 处理单篇新闻文本
│   │   ├── BatchDetection.vue   # 批量检测页面 - 同时处理多篇新闻文本
│   │   └── About.vue            # 关于系统页面 - 展示系统信息和使用说明
│   │
│   ├── router/                  # 路由配置
│   │   └── index.js             # 路由定义和导航守卫
│   │
│   ├── utils/                   # 工具类
│   │   └── api.js               # API服务封装，包含:
│   │                            # - 请求/响应拦截器
│   │                            # - API端点封装
│   │                            # - 数据格式化处理
│   │
│   ├── App.vue                  # 应用程序根组件，包含全局布局
│   └── main.js                  # 应用程序入口文件，初始化Vue实例
│
├── .env                         # 环境变量配置
├── .eslintrc.js                 # ESLint配置文件 - 代码规范检查
├── babel.config.js              # Babel配置 - JavaScript兼容性处理
├── vue.config.js                # Vue CLI配置文件 - 开发服务器和构建配置
├── package.json                 # 项目依赖配置和脚本命令
├── SUMMARY.md                   # 项目总结文档
├── start_frontend.bat           # Windows启动脚本 - 快速启动开发服务器
└── README.md                    # 项目说明文档
```

## 核心模块说明

### 组件设计

1. **ResultCard** (src/components/ResultCard.vue)
   - 功能：显示新闻检测结果，包括真伪判断、置信度和概率分布
   - 主要方法：
     - `getConfidenceValue()`: 获取正确的置信度值
     - `getPercentage()`: 将概率值转换为百分比
     - `customColorMethod()`: 根据概率值返回不同颜色

2. **LoadingIndicator** (src/components/LoadingIndicator.vue)
   - 功能：显示加载状态指示器，提供用户反馈
   - 特点：可自定义加载文本和进度

### 工具类

1. **API服务** (src/utils/api.js)
   - 功能：封装与后端API的交互
   - 主要方法：
     - `checkHealth()`: 检查API服务状态
     - `predictSingle()`: 单文本预测请求
     - `predictBatch()`: 批量文本预测请求
     - `formatSingleResponse()`: 规范化API响应数据格式

### 页面组件

1. **单文本检测** (src/views/SingleDetection.vue)
   - 功能：提供单篇新闻文本的检测界面
   - 特点：支持文本输入、示例填充、结果展示

2. **批量检测** (src/views/BatchDetection.vue)
   - 功能：同时检测多篇新闻文本
   - 特点：支持手动添加和批量粘贴两种输入方式，结果表格展示

## 功能特点

1. **单文本检测**：输入单条新闻文本，系统分析其真实性并提供详细的概率分析。
2. **批量检测**：同时分析多条新闻文本，支持手动输入和批量粘贴两种方式。
3. **结果分析**：直观展示检测结果，包括真实/虚假标签、置信度和概率分布。
4. **系统状态**：实时监控后端API和模型状态，确保系统正常运行。
5. **响应式设计**：适配不同屏幕尺寸，提供良好的移动端体验。

## 开发设置

### 安装依赖

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install
# 或者使用yarn
yarn install
```

### 启动开发服务器

```bash
# 启动开发服务器
npm run serve
# 或者使用yarn
yarn serve
```

默认情况下，开发服务器将在 http://localhost:8080 上运行。

### 构建生产版本

```bash
# 构建生产版本
npm run build
# 或者使用yarn
yarn build
```

构建后的文件将位于 `dist/` 目录中。

## 快速启动（Windows）

对于Windows用户，可以直接双击 `start_frontend.bat` 文件启动前端项目。

## API接口

前端通过以下API与后端进行交互：

- `GET /health` - 健康检查端点
- `POST /predict` - 单文本预测端点
- `POST /batch_predict` - 批量预测端点

所有API请求默认发送到 `http://localhost:5000`。可以在 `.env` 文件中通过设置 `VUE_APP_API_URL` 环境变量来修改API基础URL。

## 注意事项

1. 确保后端API服务器已经启动，前端才能正常工作。
2. 在生产环境中，您可能需要修改API的基础URL以匹配您的部署配置。

## 许可

[MIT](LICENSE) 