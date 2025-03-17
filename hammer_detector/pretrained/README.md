# 预训练模型目录

请将以下文件放置在对应目录：

1. BERT预训练模型文件：
   - 路径：`bert/`
   - 所需文件：config.json, vocab.txt, pytorch_model.bin等

2. DeiT预训练权重：
   - 文件名：`deit_base_patch16_224.pth`

3. BERT Tokenizer文件：
   - 路径：`../tokenizer/`
   - 所需文件：vocab.txt, special_tokens_map.json, tokenizer_config.json等

## 获取预训练模型

您可以从以下位置下载预训练模型：

1. BERT模型：
   - 官方地址：https://huggingface.co/bert-base-uncased
   - 您需要下载并放置在bert目录下的文件：
     - config.json
     - pytorch_model.bin
     - vocab.txt

2. DeiT预训练权重：
   - 官方地址：https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth

## 离线使用说明

如需在离线环境使用，请确保在有网络连接的环境中下载上述文件，然后将它们复制到相应目录。
