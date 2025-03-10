<template>
  <div class="home">
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="welcome-card">
          <h1>欢迎使用假新闻检测系统</h1>
          <p>本系统使用深度学习技术检测新闻文本的真伪，帮助用户识别可能的虚假信息。</p>
          <div class="features">
            <el-row :gutter="20">
              <el-col :xs="24" :sm="12" :md="8">
                <el-card shadow="hover" class="feature-card">
                  <i class="el-icon-document"></i>
                  <h3>单文本检测</h3>
                  <p>输入单条新闻文本进行真伪检测</p>
                  <el-button type="primary" @click="$router.push('/single')">开始检测</el-button>
                </el-card>
              </el-col>
              <el-col :xs="24" :sm="12" :md="8">
                <el-card shadow="hover" class="feature-card">
                  <i class="el-icon-document-copy"></i>
                  <h3>批量检测</h3>
                  <p>同时分析多条新闻文本的真伪</p>
                  <el-button type="primary" @click="$router.push('/batch')">开始检测</el-button>
                </el-card>
              </el-col>
              <el-col :xs="24" :sm="12" :md="8">
                <el-card shadow="hover" class="feature-card">
                  <i class="el-icon-picture"></i>
                  <h3>图片检测</h3>
                  <p>分析新闻图片的真实性及内容验证</p>
                  <el-button 
                    type="primary" 
                    @click="$router.push('/image-detection')"
                  >
                    开始检测
                  </el-button>
                </el-card>
              </el-col>
               <!-- 新增混合检测模块 -->
              <el-col :xs="24" :sm="12" :md="8">
                <el-card shadow="hover" class="feature-card">
                  <i class="el-icon-connection"></i>
                  <h3>混合检测</h3>
                  <p>通过URL检测多媒体新闻的综合真实性</p>
                  <el-button 
                    type="primary" 
                    @click="$router.push('/hybrid-detection')"
                  >
                    开始检测
                  </el-button>
                </el-card>
              </el-col>
              <!-- 新增视频检测模块 -->
              <el-col :xs="24" :sm="12" :md="8">
                <el-card shadow="hover" class="feature-card">
                  <i class="el-icon-video-camera"></i>
                  <h3>视频检测</h3>
                  <p>分析新闻视频的真实性及内容验证</p>
                  <el-button 
                    type="primary" 
                    @click="$router.push('/video-detection')"
                  >
                    开始检测
                  </el-button>
                </el-card>
              </el-col>
              <el-col :xs="24" :sm="12" :md="8">
                <el-card shadow="hover" class="feature-card">
                  <i class="el-icon-info"></i>
                  <h3>关于系统</h3>
                  <p>了解系统的技术细节和使用说明</p>
                  <el-button 
                    type="primary" 
                    @click="$router.push('/about')"
                  >
                    开始检测
                  </el-button>
                </el-card>
              </el-col>
            </el-row>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="20" style="margin-top: 20px;">
      <el-col :span="24">
        <el-card>
          <div slot="header" class="clearfix">
            <span>系统状态</span>
            <el-button style="float: right; padding: 3px 0" type="text" @click="checkHealth">刷新状态</el-button>
          </div>
          <div v-loading="healthLoading">
            <div v-if="healthStatus">
              <el-alert
                :title="healthStatus.status === 'ok' ? '系统正常运行中' : '系统异常'"
                :type="healthStatus.status === 'ok' ? 'success' : 'error'"
                :description="'API状态: ' + (healthStatus.status === 'ok' ? '正常' : '异常') + ', 模型已加载: ' + (healthStatus.model_loaded ? '是' : '否')"
                show-icon
              >
              </el-alert>
              <p class="timestamp">更新时间: {{ new Date(healthStatus.timestamp * 1000).toLocaleString() }}</p>
            </div>
            <div v-else>
              <el-alert
                title="未知状态"
                type="info"
                description="点击刷新按钮获取系统状态"
                show-icon
              >
              </el-alert>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
/* 新增栅格布局适配 */
.el-col-md-6 {
  width: 25%;
}

/* 保持原有样式不变 */
.feature-card {
  text-align: center;
  padding: 20px;
  margin-bottom: 20px;
  transition: transform 0.3s;
}

.feature-card:hover {
  transform: translateY(-5px);
}

.feature-card i {
  font-size: 3rem;
  color: #409EFF;
  margin-bottom: 15px;
}

.feature-card h3 {
  margin: 10px 0;
  font-size: 1.5rem;
}

.feature-card p {
  margin-bottom: 20px;
  color: #606266;
}

.timestamp {
  text-align: right;
  color: #909399;
  font-size: 0.8rem;
  margin-top: 10px;
}
</style> 