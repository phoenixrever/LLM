"""
本代码用于调用本地部署在 Ollama 上的 Mistral 模型（例如 mistral:7b-instruct-v0.3-q4_K_M），
通过 OpenAI 兼容 API 接口发送对话请求，并获取生成的回复。
"""

# 首先，安装 OpenAI 官方 Python SDK（注意：用的是 openai 库，不是 ollama）
# 终端运行： pip install openai

from openai import OpenAI  # 导入 OpenAI 客户端类（适配 OpenAI API 风格）

# 创建 OpenAI 客户端实例，连接本地 Ollama 服务
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama 默认监听的本地 API 接口
    api_key="unused",  # 虽然本地不需要 API 密钥，但 openai 库仍需要一个字符串（占位即可）
)

# 使用 chat/completions 接口发送对话请求
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",  # 设置消息角色为用户 ， "assistant"表示模型回答在多轮对话或需要记忆上下文的任务中非常有用。
            "content": [
                {
                    "type": "text",    # 内容类型为文本
                    "text": "君はだれ？君の名は？",  # 这里替换成你要输入的问题或指令
                },
            ],
        },
    ],
    model="mistral:7b-instruct-v0.3-q4_K_M",  # 指定使用的本地模型名称，要确保你已通过 Ollama 拉取并启动该模型
    max_tokens=4096,  # 最多生成的 token 数量，可根据需要调整（注意太大会影响响应速度）
    temperature=0.5,  # 控制生成内容的随机性，0.0 表示完全确定性，1.0 表示最大随机性
    top_p=0.9,  # 使用 nucleus sampling，控制生成内容的多样性
    frequency_penalty=0.0,  # 控制重复内容的惩罚，0.0 表示不惩罚
    presence_penalty=0.0,  # 控制新话题的惩罚，0.0 表示不惩罚
    # stream=False,  # 是否使用流式响应，设置为 False 表示一次性获取完整回复
    # stop=None,  # 可选的停止符号，None 表示不使用
    # user="user_id",  # 可选的用户 ID，用于跟踪请求来源（如果需要）
)

# 输出模型生成的内容
print(response.choices[0].message.content)
