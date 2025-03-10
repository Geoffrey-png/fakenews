#!/bin/bash

# 默认参数
export BATCH_SIZE=32
export LEARNING_RATE=2e-5
export WEIGHT_DECAY=0.01
export EPOCHS=3
export MAX_LEN=64
export DROPOUT_RATE=0.3
export USE_AMP=true
export SCHEDULER_TYPE=linear
export BALANCE_DATASET=true
export BALANCE_METHOD=downsample
export EARLY_STOPPING=false
export PATIENCE=3
export DELTA=0.001
export CUDA_VISIBLE_DEVICES=0
export NUM_WORKERS=4
export SEED=42

# 处理命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --batch-size=*)
      export BATCH_SIZE="${1#*=}"
      ;;
    --lr=*)
      export LEARNING_RATE="${1#*=}"
      ;;
    --weight-decay=*)
      export WEIGHT_DECAY="${1#*=}"
      ;;
    --epochs=*)
      export EPOCHS="${1#*=}"
      ;;
    --max-len=*)
      export MAX_LEN="${1#*=}"
      ;;
    --dropout=*)
      export DROPOUT_RATE="${1#*=}"
      ;;
    --amp=*)
      export USE_AMP="${1#*=}"
      ;;
    --scheduler=*)
      export SCHEDULER_TYPE="${1#*=}"
      ;;
    --balance=*)
      export BALANCE_DATASET="${1#*=}"
      ;;
    --balance-method=*)
      export BALANCE_METHOD="${1#*=}"
      ;;
    --early-stopping=*)
      export EARLY_STOPPING="${1#*=}"
      ;;
    --patience=*)
      export PATIENCE="${1#*=}"
      ;;
    --delta=*)
      export DELTA="${1#*=}"
      ;;
    --gpu=*)
      export CUDA_VISIBLE_DEVICES="${1#*=}"
      ;;
    --workers=*)
      export NUM_WORKERS="${1#*=}"
      ;;
    --seed=*)
      export SEED="${1#*=}"
      ;;
    --help)
      echo "用法: ./run_train.sh [选项]"
      echo "选项:"
      echo "  --batch-size=N    设置批次大小 (默认: 32)"
      echo "  --lr=N            设置学习率 (默认: 2e-5)"
      echo "  --weight-decay=N  设置权重衰减 (默认: 0.01)"
      echo "  --epochs=N        设置训练轮数 (默认: 3)"
      echo "  --max-len=N       设置最大序列长度 (默认: 64)"
      echo "  --dropout=N       设置Dropout率 (默认: 0.3)"
      echo "  --amp=BOOL        是否使用混合精度训练 (默认: true)"
      echo "  --scheduler=TYPE  设置学习率调度器类型 (linear/cosine/constant, 默认: linear)"
      echo "  --balance=BOOL    是否平衡数据集 (默认: true)"
      echo "  --balance-method=M 平衡方法 (downsample/upsample, 默认: downsample)"
      echo "  --early-stopping=B 是否使用早停 (默认: false)"
      echo "  --patience=N      早停耐心值 (默认: 3)"
      echo "  --delta=N         早停阈值 (默认: 0.001)"
      echo "  --gpu=IDS         使用的GPU ID (默认: 0)"
      echo "  --workers=N       数据加载的工作线程数 (默认: 4)"
      echo "  --seed=N          随机种子 (默认: 42)"
      echo "  --help            显示此帮助信息并退出"
      exit 0
      ;;
    *)
      echo "未知选项: $1"
      echo "使用 --help 查看帮助信息"
      exit 1
      ;;
  esac
  shift
done

# 验证参数的正确性
if [[ ! "$BALANCE_METHOD" =~ ^(downsample|upsample|weight)$ ]]; then
  echo "错误: 平衡方法必须是 'downsample'、'upsample' 或 'weight'"
  exit 1
fi

if [[ ! "$SCHEDULER_TYPE" =~ ^(linear|cosine|constant)$ ]]; then
  echo "错误: 学习率调度器类型必须是 'linear'、'cosine' 或 'constant'"
  exit 1
fi

# 打印参数
echo "训练参数:"
echo "批次大小: $BATCH_SIZE"
echo "学习率: $LEARNING_RATE"
echo "权重衰减: $WEIGHT_DECAY"
echo "训练轮数: $EPOCHS"
echo "最大序列长度: $MAX_LEN"
echo "Dropout率: $DROPOUT_RATE"
echo "使用混合精度: $USE_AMP"
echo "学习率调度器: $SCHEDULER_TYPE"
echo "平衡数据集: $BALANCE_DATASET"
echo "平衡方法: $BALANCE_METHOD"
echo "使用早停: $EARLY_STOPPING"
echo "早停耐心值: $PATIENCE"
echo "早停阈值: $DELTA"
echo "GPU IDs: $CUDA_VISIBLE_DEVICES"
echo "工作线程数: $NUM_WORKERS"
echo "随机种子: $SEED"

# 修改配置文件
cat > src/runtime_config.py << EOF
import os
import torch

# 从环境变量读取配置
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 2e-5))
WEIGHT_DECAY = float(os.environ.get('WEIGHT_DECAY', 0.01))
EPOCHS = int(os.environ.get('EPOCHS', 3))
MAX_LEN = int(os.environ.get('MAX_LEN', 64))
DROPOUT_RATE = float(os.environ.get('DROPOUT_RATE', 0.3))
USE_AMP = os.environ.get('USE_AMP', 'true').lower() == 'true'
SCHEDULER_TYPE = os.environ.get('SCHEDULER_TYPE', 'linear')
BALANCE_DATASET = os.environ.get('BALANCE_DATASET', 'true').lower() == 'true'
BALANCE_METHOD = os.environ.get('BALANCE_METHOD', 'downsample')
EARLY_STOPPING = os.environ.get('EARLY_STOPPING', 'false').lower() == 'true'
PATIENCE = int(os.environ.get('PATIENCE', 3))
DELTA = float(os.environ.get('DELTA', 0.001))
CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 4))
SEED = int(os.environ.get('SEED', 42))

# 从基础配置导入其他设置
from config import *
EOF

# 启动训练
echo "开始训练..."
cd src
python -c "import runtime_config as config; print('配置已加载'); from train_with_config import main; main()"

echo "训练完成！" 