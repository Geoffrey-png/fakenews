# HAMMER - 多模态篡改检测工具

HAMMER (Hierarchical Attentional Multi-modal rEpResentation leaRning) 是一个强大的多模态篡改检测工具，能够检测图像和文本中的伪造内容。

## 功能特点

- 支持多种图像篡改检测（换脸、面部属性修改、物体添加/删除/修改等）
- 支持文本属性篡改检测
- 提供篡改区域定位（边界框）
- 提供篡改类型识别
- 支持批量处理
- 支持离线使用

## 安装说明

### 环境需求

- Python 3.6+
- PyTorch 1.7+
- CUDA (如需GPU加速)

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/username/hammer-detector.git
cd hammer-detector
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 设置预训练模型结构：

```bash
python setup_pretrained.py
```

4. 准备预训练模型文件：

在有网络的环境中，下载以下预训练模型文件，并将它们放置在对应目录：

- HAMMER模型权重：
  - 放置位置：`weights/checkpoint_best.pth`

- DeiT预训练权重：
  - 下载地址：https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
  - 放置位置：`pretrained/deit_base_patch16_224.pth`

- BERT预训练模型文件：
  - 下载地址：https://huggingface.co/bert-base-uncased
  - 所需文件：config.json, vocab.txt, pytorch_model.bin等
  - 放置位置：`pretrained/bert/`

- BERT Tokenizer文件：
  - 下载地址：https://huggingface.co/bert-base-uncased
  - 所需文件：vocab.txt, special_tokens_map.json, tokenizer_config.json等
  - 放置位置：`tokenizer/`

## 离线使用指南

本工具可以在离线环境中使用，无需网络连接。按照以下步骤准备：

1. 在有网络的环境中，运行设置脚本：

```bash
python setup_pretrained.py
```

2. 下载所有所需的预训练模型文件，可以使用 huggingface-cli 等工具：

```bash
# 下载BERT模型
pip install huggingface-hub
huggingface-cli download bert-base-uncased --local-dir ./pretrained/bert

# 下载BERT Tokenizer
huggingface-cli download bert-base-uncased --local-dir ./tokenizer
```

3. 下载DeiT预训练权重：

```bash
wget https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth -P ./pretrained/
```

4. 将下载的所有文件转移到离线环境中的相应目录。

5. 确认文件结构正确：

```
hammer_detector/
├── weights/
│   └── checkpoint_best.pth
├── pretrained/
│   ├── bert/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   └── deit_base_patch16_224.pth
└── tokenizer/
    ├── vocab.txt
    ├── special_tokens_map.json
    └── tokenizer_config.json
```

6. 运行检查脚本，确认模型文件都已就绪：

```bash
python check_weights.py
```

## 使用方法

### 单张图像检测

使用`simple_detect.py`脚本可以检测单张图像：

```bash
python simple_detect.py --image path/to/image.jpg
```

可选参数：
- `--text` - 与图像相关的文本（如有）

### 检测结果

检测结果将包含以下信息：

- `is_fake` - 图像是否被篡改的布尔值
- `forgery_score` - 篡改概率分数（0-1之间）
- `manipulation_type` - 篡改类型（如"换脸"、"面部属性修改"等）
- `bboxes` - 篡改区域位置的边界框坐标（如有）

结果将保存为JSON文件，格式如下：

```json
{
  "is_fake": true,
  "forgery_score": 0.85,
  "manipulation_type": "换脸",
  "bboxes": [[100, 150, 300, 350]]
}
```

## 批量检测

使用`batch_detect.py`脚本可以批量处理多张图像：

```bash
python batch_detect.py --image_dir path/to/images --output output.json
```

## API使用

您也可以在自己的Python代码中导入和使用HAMMER检测器：

```python
from models.detector import HAMMERDetector

# 初始化检测器
detector = HAMMERDetector(
    config_path="config.yaml",
    model_path="weights/checkpoint_best.pth"
)

# 检测单张图像
result = detector.detect(image_path="path/to/image.jpg")
print(f"篡改概率: {result['forgery_score']}")
print(f"篡改类型: {result['manipulation_type']}")

# 批量检测
results = detector.detect_batch(image_paths=["image1.jpg", "image2.jpg"])
```

## 故障排除

### 中文显示问题

在Windows命令行环境下，中文可能会显示为乱码。这是由于编码问题导致的，但不影响检测功能和结果保存。JSON文件是正确编码的，可以通过其他工具（如文本编辑器）正确查看。

### 内存问题

如果遇到内存不足的情况，可以尝试以下方法：
1. 使用较小分辨率的输入图像
2. 在CPU模式下运行（设置`CUDA_VISIBLE_DEVICES=-1`环境变量）

### 离线使用问题

1. 确保所有预训练模型文件都已正确放置：
   - 模型权重文件 `weights/checkpoint_best.pth`
   - BERT模型文件在 `pretrained/bert/` 目录
   - DeiT预训练权重 `pretrained/deit_base_patch16_224.pth`
   - BERT分词器文件在 `tokenizer/` 目录

2. 如果缺少某些文件，程序会给出具体的错误提示和文件路径
3. 可以通过运行 `setup_pretrained.py` 创建正确的目录结构
4. 对于特殊字符或中文内容的处理，确保所有文件使用UTF-8编码

## 参考文献

如需了解更多技术细节，请参考以下论文：

[HAMMER: Multimodal Forgery Detection with Hierarchical Attention Networks](https://arxiv.org/abs/xxxx.xxxx)

## 许可证

本项目基于MIT许可证开源。 