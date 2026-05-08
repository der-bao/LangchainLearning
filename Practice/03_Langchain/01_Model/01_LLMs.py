from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv
import os

load_dotenv()

model = Tongyi(model = "qwen-max", api_key = os.getenv("LLM_API_KEY"))

# # 调用 invoke 向模型提问 - 直接输出
# res = model.invoke(input="你是谁？")
# print(type(res))
# print(res)

"""
# output
<class 'str'>
我是Qwen，由阿里云开发的预训练语言模型。我被设计用来帮助用户生成各种类型的文本，
如文章、故事、诗歌、故事等，并能够根据不同的场景和需求提供信息和帮助。
此外，我还能够回答问题、提供解释、进行对话等。有什么我可以帮到你的吗？
"""

# 调用 stream 方法向模型提问 - 流式输出
res = model.stream(input="什么是人工智能？")
# print(type(res))                        # <class 'generator'>，说明 res 是一个生成器对象，可以使用 for 循环来迭代获取每个输出块
for chunk in res:
    # print(type(chunk))                # str
    # if not chunk:
    #     print("chunk is empty")       # 可能会有空字符串的情况，导致输出时没有内容
    # print(chunk)
    print(chunk, end="", flush=True)