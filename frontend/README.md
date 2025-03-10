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
├── public/                 # 静态资源
│   └── index.html          # HTML入口文件
├── src/                    # 源代码
│   ├── assets/             # 图片等静态资源
│   │   └── styles.css      # 全局样式
│   ├── components/         # 公共组件
│   │   ├── LoadingIndicator.vue  # 加载指示器组件
│   │   └── ResultCard.vue        # 结果展示卡片组件
│   ├── views/              # 页面组件
│   │   ├── Home.vue        # 首页
│   │   ├── SingleDetection.vue  # 单文本检测页面
│   │   ├── BatchDetection.vue   # 批量检测页面
│   │   └── About.vue       # 关于系统页面
│   ├── router/             # 路由配置
│   │   └── index.js        # 路由定义
│   ├── utils/              # 工具类
│   │   └── api.js          # API服务
│   ├── App.vue             # 应用程序入口组件
│   └── main.js             # 应用程序入口文件
├── babel.config.js         # Babel配置
├── vue.config.js           # Vue配置文件
├── package.json            # 项目依赖
├── start_frontend.bat      # Windows启动脚本
└── README.md               # 项目说明
```

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