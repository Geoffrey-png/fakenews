# 中文假新闻检测系统

本项目是一个基于BERT的中文假新闻检测系统，能够自动识别新闻文本是真实新闻还是虚假新闻（谣言）。该系统使用了`bert-base-chinese`预训练模型，并通过微调使其适应假新闻检测任务。

## 项目结构

```
fakenews/
├── data/                 # 数据目录
│   ├── train.xlsx        # 训练数据
│   ├── val.xlsx          # 验证数据
│   └── test.xls          # 测试数据
├── models/               # 保存的模型
├── logs/                 # 训练日志
├── report/               # 评估报告
├── src/                  # 源代码
│   ├── config.py         # 配置文件
│   ├── data_processor.py # 数据处理
│   ├── train_with_config.py   # 训练脚本
│   ├── evaluate_model.py # 评估脚本
│   └── generate_report.py # 报告生成
└── run_train.sh          # 训练启动脚本
```

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- Transformers 4.0+
- pandas, numpy, scikit-learn, matplotlib, seaborn
- jieba (中文分词)
- xlrd, openpyxl (Excel文件处理)

## 安装步骤

1. 克隆仓库：
   ```bash
   git clone [仓库URL]
   cd fakenews
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 使用配置脚本训练模型

使用提供的shell脚本可以轻松调整训练参数并运行训练：

```bash
chmod +x run_train.sh  # 确保脚本有执行权限
./run_train.sh --batch-size=32 --lr=2e-5 --epochs=3
```

#### 可用参数：

- `--batch-size=N` - 设置批次大小 (默认: 32)
- `--lr=N` - 设置学习率 (默认: 2e-5)
- `--weight-decay=N` - 设置权重衰减 (默认: 0.01)
- `--epochs=N` - 设置训练轮数 (默认: 3)
- `--max-len=N` - 设置最大序列长度 (默认: 64)
- `--dropout=N` - 设置Dropout率 (默认: 0.3)
- `--amp=BOOL` - 是否使用混合精度训练 (默认: true)
- `--scheduler=TYPE` - 学习率调度器类型 (linear/cosine/constant, 默认: linear)
- `--balance=BOOL` - 是否平衡数据集 (默认: true)
- `--balance-method=M` - 平衡方法 (downsample/upsample, 默认: downsample)
- `--early-stopping=BOOL` - 是否使用早停 (默认: false)
- `--patience=N` - 早停耐心值 (默认: 3)
- `--delta=N` - 早停阈值 (默认: 0.001)
- `--gpu=IDS` - 使用的GPU ID (默认: 0)
- `--workers=N` - 数据加载的工作线程数 (默认: 4)
- `--seed=N` - 随机种子 (默认: 42)

### 直接修改配置文件

也可以直接编辑`src/config.py`文件来修改训练参数，然后运行：

```bash
cd src
python train_with_config.py
```

### 评估模型

要评估已训练的模型，可以运行：

```bash
cd src
python evaluate_model.py
```

### 生成评估报告

要生成模型评估的详细报告，可以运行：

```bash
cd src
python generate_report.py
```
