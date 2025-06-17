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


def build_production_prompt(java_code: str):
    return [
        {
            "role": "system",
            "content": "你是一个资深 Java 工程师，擅长为 Java 代码生成高质量的 JUnit 5 测试用例。请根据用户提供的 Java 方法或类，生成对应的 JUnit 5 测试类，遵循良好的测试实践，包含常规情况、边界条件和可能的异常处理测试。",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"请为以下 Java 类或方法生成 JUnit 5 测试代码：\n\n```java\n{java_code}\n```",
                }
            ],
        },
    ]


# Python 是解释型语言，代码是从上往下依次执行的。
# 当第二个 say_hello() 被定义时，它会覆盖之前那个函数的定义。
# 所以最终程序里只有一个 say_hello()，就是最后一个版本。


"""
  异常处理测试

  边界条件测试

  使用 Mockito Mock 依赖

  遵循 AAA（Arrange-Act-Assert）结构

"""


def build_production_prompt(java_code: str):
    return [
        {
            "role": "system",
            "content": (
                "あなたは経験豊富なJavaエンジニアです。"
                "ユーザーが提供したJavaクラスまたはメソッドに対して、JUnit 5のテストコードを高品質かつベストプラクティスに従って作成してください。"
                "通常のケースだけでなく、例外処理や境界値のテストも含めてください。"
                "依存関係があればMockitoを使ってモックしてください。"
                "テストコードはArrange-Act-Assert（AAA）構造に従ってください。"
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"以下のJavaコードに対して、JUnit 5のテストコードを生成してください：\n\n```java\n{java_code}\n```",
                }
            ],
        },
    ]


# 英文>> 中文 >> 日文


"""
You are a helpful assistant specialized in Java unit testing.


You are an experienced Java engineer. Please generate high-quality JUnit 5 test code for the following Java class or method.
- Include normal cases, exception handling tests, and boundary tests.
- Use Mockito to mock dependencies if any.
- Follow the Arrange-Act-Assert (AAA) testing pattern.
- Write all code comments and explanations in Japanese.
Here is the Java code:\n\n" f"```java\n{{java_code}} \n```


public class Calculator {
public int add(int a, int b) {
    return a + b;
}

public int divide(int a, int b) {
    if (b == 0) {
        throw new IllegalArgumentException("除数不能为0");
    }
    return a / b;
}
}
"""


def build_production_prompt(java_code: str):
    # 构造 User prompt
    user_prompt = (
        "You are an experienced Java engineer. Please generate high-quality JUnit 5 test code for the following Java class or method.\n"
        "- Include normal cases, exception handling tests, and boundary tests.\n"
        "- Use Mockito to mock dependencies if any.\n"
        "- Follow the Arrange-Act-Assert (AAA) testing pattern.\n"
        "- Write all code comments and explanations in Japanese.\n\n"
        "Here is the Java code:\n\n"
        f"```java\n{java_code}  \n```"
    )
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant specialized in Java unit testing.",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        },
    ]


java_code = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int divide(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("除数不能为0");
        }
        return a / b;
    }
}
"""
messages = build_production_prompt(java_code)

# 使用 chat/completions 接口发送对话请求
response = client.chat.completions.create(
    messages=messages,
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
