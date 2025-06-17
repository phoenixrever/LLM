from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True).cuda()

# 输入提示语，例如：要求写一个快速排序算法
input_text = "#write a quick sort algorithm"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

# 生成代码
outputs = model.generate(**inputs, max_length=128)

# 打印模型生成的内容
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
