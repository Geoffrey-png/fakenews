<template>
  <div class="image-detection">
    <el-card class="upload-card">
      <div slot="header">
        <span>图片新闻检测</span>
      </div>
      <el-upload
        class="upload-area"
        drag
        action="/api/upload"
        :before-upload="beforeUpload"
        :on-success="handleSuccess"
        :on-error="handleError"
        :show-file-list="false"
      >
        <i class="el-icon-upload"></i>
        <div class="el-upload__text">将图片拖到此处，或<em>点击上传</em></div>
        <div class="el-upload__tip" slot="tip">
          支持格式：JPEG/PNG，大小不超过5MB
        </div>
      </el-upload>

      <el-alert
        v-if="result"
        :title="result.title"
        :type="result.type"
        :description="result.description"
        show-icon
        class="result-alert"
      />

      <div class="preview-area" v-if="imageUrl">
        <el-image 
          :src="imageUrl"
          fit="contain"
          style="max-height: 400px;"
        >
          <div slot="placeholder" class="image-placeholder">
            加载中...
          </div>
        </el-image>
      </div>
      <!-- 修改跳转路径 -->
      <el-button 
        type="primary" 
        @click="$router.push('/image-detection')">
        开始检测
      </el-button>

    </el-card>
  </div>
</template>

<script>
export default {
  name: 'ImageDetection',
  data() {
    return {
      imageUrl: '',
      result: null,
      loading: false
    }
  },
  methods: {
    beforeUpload(file) {
      const isImage = ['image/jpeg', 'image/png'].includes(file.type)
      const isLt5M = file.size / 1024 / 1024 < 5

      if (!isImage) {
        this.$message.error('只能上传 JPG/PNG 格式的图片!')
      }
      if (!isLt5M) {
        this.$message.error('图片大小不能超过5MB!')
      }
      return isImage && isLt5M
    },

    handleSuccess(response, file) {
      this.imageUrl = URL.createObjectURL(file.raw)
      // 调用检测API
      this.$api.uploadImage(file.raw)
        .then(res => {
          this.result = {
            title: res.data.is_fake ? '疑似虚假新闻' : '可信新闻',
            type: res.data.is_fake ? 'error' : 'success',
            description: `可信度评分: ${(res.data.confidence * 100).toFixed(1)}%`
          }
        })
        .catch(error => {
          this.$message.error('检测失败: ' + error.message)
        })
    },

    handleError() {
      this.$message.error('上传失败，请重试')
    }
  }
}
</script>

<style scoped>
.upload-card {
  max-width: 800px;
  margin: 20px auto;
}

.upload-area {
  margin: 20px 0;
}

.preview-area {
  margin-top: 30px;
  text-align: center;
}

.result-alert {
  margin: 20px 0;
  font-size: 16px;
}

.image-placeholder {
  padding: 50px 0;
}
</style>
