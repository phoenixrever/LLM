
# 从 Hugging Face Transformers 库导入 BERT 专用的分词器和文本分类模型类
import os
from transformers import BertTokenizer, BertForSequenceClassification

# 导入 pipeline 工具，用于快速构建高层次 NLP 应用（如文本分类、情感分析等）
from transformers import pipeline

# model_name 是本地下载或微调后的模型所在的路径
# 路径可以是远程模型名（如 "bert-base-chinese"），也可以是本地目录
model_path=os.path.join(os.path.dirname(__file__), "model/bert-base-chinese/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")

# 加载 BERT 中文分词器（Tokenizer）
# Tokenizer 负责把自然语言（如中文）转换为模型输入所需的 token ID（数字序列）
# 它会自动加载 vocab.txt、tokenizer_config.json 等文件
tokenizer = BertTokenizer.from_pretrained(model_path)

# 加载 BERT 模型并指定为“文本分类”任务（Sequence Classification）
# 这个模型默认结构是一个 Transformer + 一个分类头（线性层）
# 会自动加载 config.json 和 pytorch_model.bin 等文件
model = BertForSequenceClassification.from_pretrained(model_path)



'''
| 任务类型                             | 描述                 |
| -------------------------------- | ------------------ |
| `"text-classification"`          | 文本分类，比如情感分析        |
| `"token-classification"`         | 序列标注（如命名实体识别）      |
| `"translation"`                  | 翻译                 |
| `"summarization"`                | 文本摘要               |
| `"question-answering"`           | 问答系统               |
| `"text-generation"`              | 文本生成（例如 GPT 模型）    |
| `"zero-shot-classification"`     | 零样本分类              |
| `"conversational"`               | 对话建模               |
| `"feature-extraction"`           | 特征提取               |
| `"fill-mask"`                    | 填空任务（如 BERT）       |
| `"image-classification"`         | 图像分类（需要 vision 模型） |
| `"automatic-speech-recognition"` | 语音识别（ASR）          |

'''

# 使用 Hugging Face 的 pipeline 创建一个文本分类器
# pipeline 是高阶封装工具，可以一行代码完成 tokenization + 推理 + 解码
# "text-classification" 指定任务类型
# model 和 tokenizer 是我们刚刚加载的实例
# device="cuda" 表示使用 GPU 推理（需确保 CUDA 驱动和 torch 支持 GPU）
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cuda")

# 使用分类器对一句话进行分类（如情感分析、类别识别）
result = classifier("你好，我是一款语言模型")

# 打印结果，格式可能是：
# [{'label': 'LABEL_0', 'score': 0.987654}]
print(result)

# 打印模型结构信息（可选）
# 会显示模型的层级、参数结构等
print(model)

"""
| 名称                             | 类型    | 说明                                                                                 |
| ------------------------------- | ----    | ---------------------------------------------------------------------------------- |
| `BertTokenizer`                 | 类      | 基于 BERT 的分词器，用于将文本转为 token ID。支持中文、英文等，处理方式如：将句子变成 `[CLS] 你好 我 是 一款 [SEP]` 并转为 ID。 |
| `BertForSequenceClassification` | 类      | 基于 BERT 的文本分类模型。内部结构是 `BertModel + 分类头`，适用于情感分析、垃圾识别、文本归类等任务。                      |
| `pipeline`                      | 函数     | Hugging Face 提供的“快捷工具函数”，用于快速构建常见任务（如文本生成、分类、翻译、问答等），你只需指定任务名，剩下交给它处理。             |
| `classifier()`                  | 函数调用 | 对输入文本进行推理分类，自动完成分词、张量化、模型推理、后处理。                                                   |

"""