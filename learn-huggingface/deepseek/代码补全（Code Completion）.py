# 我运行不了看看就好

# 导入 Hugging Face Transformers 库中的两个主要类：
# AutoTokenizer：自动加载与指定模型对应的分词器（Tokenizer），负责将文本转为模型可接受的输入格式（token IDs）
# AutoModelForCausalLM：自动加载用于因果语言建模的预训练模型（适合文本生成任务）
from transformers import AutoTokenizer, AutoModelForCausalLM


# 默认缓存路径 Windows: C:\Users\<你的用户名>\.cache\huggingface\transformers\

cache_dir="D:\code\LLM"

# 加载预训练的分词器（Tokenizer）
# 参数 "deepseek-ai/deepseek-coder-1.3b-base" 指定了模型名称或路径
# trust_remote_code=True 允许加载模型作者自定义的代码（比如特殊 tokenizer 或模型结构）
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base",trust_remote_code=True, cache_dir=cache_dir)

# 加载预训练的因果语言模型（Causal LM）
# 并将模型移动到GPU上（如果有CUDA支持）
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base",trust_remote_code=True).cuda()

# 定义输入提示文本，告诉模型要生成的内容，这里是“写一个快速排序算法”
input_text = "#write a quick sort algorithm"

# 使用 tokenizer 将输入文本编码成模型所需的张量格式
# return_tensors="pt" 指定返回 PyTorch 张量格式
# .to(model.device) 将输入张量移动到模型所在设备（GPU或CPU）
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# 使用模型生成文本
# generate() 是 Transformers 中用于文本生成的方法
# 参数 max_length=128 指定生成文本的最大长度（包含输入长度）
outputs = model.generate(
  **inputs,
  do_sample=True,  # 启用采样，增加生成文本的多样性
  temperature=0.7,  # 控制生成文本的随机性，值越低越确定，值越高越随机
  top_p=0.9,  # nucleus sampling，控制生成文本的多样性
  top_k=50,  # 控制生成文本的多样性，限制每
  # 次生成时考虑的最高概率的 token 数量
  repetition_penalty=1.2,  # 重复惩罚，防止模型生成重复的内容
  # num_return_sequences=1,  # 返回生成的序列数量，这里只返回一个序列
  # pad_token_id=tokenizer.eos_token_id,  # 填充 token ID 使用模型的结束 token ID 作为填充 token ID 这样可以确保生成的文本不会因为填充而被截断
  # use_cache=True,  # 启用缓存以加速生成过程 在生成过程中使用缓存，减少计算量 适用于长文本生成任务
  # 这里是为了提高生成效率，避免不必要的计算  例如，当生成的文本达到某个长度或满足某个条件时停止
  early_stopping=True,  # 提前停止生成，当达到某个条件时停止  
  max_length=128)

# 解码生成的 token IDs 为字符串
# skip_special_tokens=True 表示跳过特殊符号（如 <pad>、<eos> 等）
#  **inputs 是 Python 的解包操作符，**inputs 会把字典里的键值对拆开，作为关键字参数传给函数。
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
