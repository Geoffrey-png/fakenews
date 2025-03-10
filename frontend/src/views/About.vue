<template>
  <div class="about">
    <div class="page-header">
      <h1>关于系统</h1>
      <p>了解假新闻检测系统的技术细节和使用说明</p>
    </div>

    <el-card class="info-card">
      <el-tabs>
        <el-tab-pane label="系统简介">
          <div class="tab-content">
            <h2>假新闻检测系统</h2>
            <p>假新闻检测系统是一个基于深度学习技术的自然语言处理应用，旨在帮助用户识别可能的虚假新闻。系统通过分析新闻文本的语言特征、内容特征和上下文信息，判断新闻的真实性。</p>
            
            <h3>主要功能</h3>
            <el-divider></el-divider>
            <div class="features">
              <div class="feature-item">
                <i class="el-icon-document"></i>
                <div class="feature-info">
                  <h4>单文本检测</h4>
                  <p>分析单条新闻文本的真实性，提供详细的概率分析和置信度。</p>
                </div>
              </div>
              <div class="feature-item">
                <i class="el-icon-document-copy"></i>
                <div class="feature-info">
                  <h4>批量检测</h4>
                  <p>同时处理多条新闻文本，适合需要进行大规模新闻真伪判断的场景。</p>
                </div>
              </div>
              <div class="feature-item">
                <i class="el-icon-data-analysis"></i>
                <div class="feature-info">
                  <h4>结果分析</h4>
                  <p>提供直观的检测结果展示和统计分析，便于用户理解和使用。</p>
                </div>
              </div>
            </div>
          </div>
        </el-tab-pane>
        
        <el-tab-pane label="技术架构">
          <div class="tab-content">
            <h2>系统架构</h2>
            <p>假新闻检测系统采用前后端分离的架构设计，前端使用Vue.js和Element UI构建用户界面，后端基于Flask提供REST API服务。</p>
            
            <div class="architecture">
              <div class="arch-section">
                <h3>前端技术栈</h3>
                <ul>
                  <li><strong>Vue.js:</strong> 渐进式JavaScript框架</li>
                  <li><strong>Element UI:</strong> 基于Vue.js的组件库</li>
                  <li><strong>Axios:</strong> 基于Promise的HTTP客户端</li>
                  <li><strong>Vue Router:</strong> Vue.js官方路由管理器</li>
                </ul>
              </div>
              
              <div class="arch-section">
                <h3>后端技术栈</h3>
                <ul>
                  <li><strong>Flask:</strong> 轻量级Web应用框架</li>
                  <li><strong>PyTorch:</strong> 深度学习框架</li>
                  <li><strong>Transformers:</strong> 自然语言处理模型库</li>
                  <li><strong>BERT:</strong> 预训练语言模型</li>
                </ul>
              </div>
            </div>
            
            <h3>模型简介</h3>
            <p>系统的核心是一个基于BERT的深度学习模型，通过对大量真实和虚假新闻的学习，能够识别新闻文本中的真实性特征。模型经过调优，在测试集上达到了较高的准确率。</p>
          </div>
        </el-tab-pane>
        
        <el-tab-pane label="使用指南">
          <div class="tab-content">
            <h2>使用指南</h2>
            <p>以下是系统各功能的详细使用说明，帮助用户快速上手使用假新闻检测系统。</p>
            
            <div class="guide-section">
              <h3>单文本检测</h3>
              <ol>
                <li>导航到"单文本检测"页面</li>
                <li>在文本框中输入或粘贴需要检测的新闻文本</li>
                <li>点击"开始检测"按钮</li>
                <li>系统将分析文本并显示检测结果</li>
                <li>查看结果中的标签、置信度和概率分布</li>
              </ol>
              <p class="tip">提示：可以使用"填充示例"按钮快速测试系统功能。</p>
            </div>
            
            <div class="guide-section">
              <h3>批量检测</h3>
              <ol>
                <li>导航到"批量检测"页面</li>
                <li>选择输入方式：手动输入或批量粘贴</li>
                <li>输入多条需要检测的新闻文本</li>
                <li>点击"开始批量检测"按钮</li>
                <li>系统将分析所有文本并显示检测结果列表</li>
                <li>可以查看详情、排序或导出结果</li>
              </ol>
              <p class="tip">提示：批量检测适合同时处理多条新闻，提高检测效率。</p>
            </div>
          </div>
        </el-tab-pane>
        
        <el-tab-pane label="注意事项">
          <div class="tab-content">
            <h2>使用注意事项</h2>
            <p>为了获得更准确的检测结果，使用系统时请注意以下几点：</p>
            
            <div class="notice-section">
              <h3>系统局限性</h3>
              <ul>
                <li>模型基于训练数据进行判断，对于全新类型的虚假新闻可能存在误判</li>
                <li>系统主要分析文本内容，无法验证新闻中提及的外部事实</li>
                <li>对于极短的文本（少于10个字符）检测效果可能不佳</li>
                <li>系统不能替代人工判断，仅作为辅助工具使用</li>
              </ul>
            </div>
            
            <div class="notice-section">
              <h3>最佳实践</h3>
              <ul>
                <li>提供完整的新闻文本，包含标题和正文，以提高检测准确率</li>
                <li>批量检测时，每条新闻文本之间应明确分隔</li>
                <li>检测结果的置信度越高，可信度越大</li>
                <li>对于置信度低于70%的结果，建议结合其他信息进行综合判断</li>
              </ul>
            </div>
            
            <el-alert
              title="免责声明"
              type="warning"
              description="本系统仅作为技术演示和辅助工具，不对检测结果的准确性做任何保证。用户在使用系统时应自行承担风险，系统开发者不对使用结果承担任何法律责任。"
              show-icon
              :closable="false"
            >
            </el-alert>
          </div>
        </el-tab-pane>
      </el-tabs>
    </el-card>
  </div>
</template>

<script>
export default {
  name: 'About'
}
</script>

<style scoped>
.about {
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

.info-card {
  margin-bottom: 30px;
}

.tab-content {
  padding: 10px;
}

.tab-content h2 {
  color: #409EFF;
  border-bottom: 2px solid #EBEEF5;
  padding-bottom: 10px;
  margin-top: 0;
}

.tab-content h3 {
  color: #303133;
  margin: 20px 0 10px;
}

.features {
  margin-top: 20px;
}

.feature-item {
  display: flex;
  margin-bottom: 20px;
  align-items: flex-start;
}

.feature-item i {
  font-size: 2rem;
  color: #409EFF;
  margin-right: 15px;
  margin-top: 5px;
}

.feature-info h4 {
  margin: 0 0 5px;
  color: #303133;
}

.feature-info p {
  margin: 0;
  color: #606266;
}

.architecture {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin: 20px 0;
}

.arch-section {
  flex: 1;
  min-width: 300px;
  background-color: #f9f9f9;
  padding: 15px;
  border-radius: 4px;
}

.arch-section h3 {
  margin-top: 0;
  color: #303133;
  border-bottom: 1px solid #EBEEF5;
  padding-bottom: 10px;
}

.arch-section ul {
  padding-left: 20px;
}

.arch-section li {
  margin-bottom: 8px;
  color: #606266;
}

.guide-section, .notice-section {
  margin-bottom: 30px;
}

.guide-section h3, .notice-section h3 {
  color: #303133;
  margin-bottom: 15px;
}

.guide-section ol, .notice-section ul {
  padding-left: 20px;
  color: #606266;
}

.guide-section li, .notice-section li {
  margin-bottom: 10px;
}

.tip {
  background-color: #ecf8ff;
  padding: 8px 16px;
  border-radius: 4px;
  border-left: 5px solid #50bfff;
  margin: 20px 0;
  color: #5e6d82;
}

.el-alert {
  margin-top: 30px;
}
</style> 