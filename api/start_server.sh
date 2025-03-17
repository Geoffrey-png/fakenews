#!/bin/bash

# 设置环境变量（可以根据需要修改）
export API_HOST=0.0.0.0
export API_PORT=5000
export DEBUG=false
export MAX_SEQUENCE_LENGTH=128
export BATCH_SIZE=8

# 创建日志目录
mkdir -p ../logs

echo "启动假新闻检测API服务..."

# 检查是否存在gunicorn
if command -v gunicorn &> /dev/null; then
    echo "使用gunicorn启动服务（生产模式）"
    gunicorn -w 4 -b ${API_HOST}:${API_PORT} app:app
else
    echo "gunicorn未安装，使用Flask开发服务器启动"
    python app.py
fi 