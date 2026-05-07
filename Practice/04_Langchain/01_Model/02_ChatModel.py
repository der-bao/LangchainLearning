from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("LLM_MODEL_ID"),
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL")
)

# messages = [
#     SystemMessage(content="你是我的人工智能助手，协助我解答问题。"),
#     HumanMessage(content="请介绍一下自己。")
# ]

# 也可以直接用字典列表的形式，Langchain会自动转换成对应的消息对象
messages = [
    {"role": "system", "content": "你是我的人工智能助手，协助我解答问题。"},
    {"role": "user", "content": "请介绍一下自己。"}
]


# 调用 invoke - 直接输出
# response = model.invoke(messages)
# print(type(response))               # <class 'langchain_core.messages.ai.AIMessage'>
# print(response)                     # 调用继承于BaseMessage的__str__方法，输出消息内容

# AIMessage → BaseMessage → Serializable → BaseModel

# 调用 stream - 流式输出
response_stream = model.stream(messages)
print(type(response_stream))        # <class 'generator'>
for chunk in response_stream:
    # print(type(chunk))              # <class 'langchain_core.messages.ai.AIMessageChunk'>
    print(chunk.content,end="",flush=True)                    # 每次迭代输出一个消息块，直到流结束

# AIMessageChunk → AIMessage，BaseMessageChunk → BaseMessage → Serializable → BaseModel