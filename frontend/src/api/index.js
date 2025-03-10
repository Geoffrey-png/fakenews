// ... 已有API配置 ...
uploadImage(file) {
  const formData = new FormData()
  formData.append('image', file)
  return instance.post('/detect/image', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}
// ... 剩余配置 ...
