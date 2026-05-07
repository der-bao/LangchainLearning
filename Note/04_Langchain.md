# Langchain

## 一、 简介

**LangChain** 是一个用于构建基于大语言模型 (LLM) 的应用框架，它可以方便地将 LLM 与外部数据源、工具、记忆系统结合，实现问答、对话、自动化脚本等功能。

**核心架构六大组件**:

- Models：LLM 模型、Chat 模型、嵌入模型
- Prompts：提示词模板、FewShot 示例
- Documents：文档加载器、文本分割器
- Memory：会话记忆、向量记忆、实体记忆
- Chains：任务链路编排（问答链、摘要链、路由链）
- Agents：智能体 + 工具调用，自主决策执行任务

## 二、安装部署

```
pip install langchain
pip install langchain-community         # 社区版，支持第三方模型调用
pip install langchain-ollama            # 支持调用ollama本地部署的llm
```

## 三、核心模块

### 1. Model

Langchain支持三种模型，分别是LLMs、Chat Model、Embedding Model。

#### (1) LLMs
该类模型是**文本补全**模型
输入和输出都是**字符串**
参考示例如下：

```
from langchain_community.llms.tongyi import Tongyi
from dotenv import load_dotenv
import os

load_dotenv()

model = Tongyi(model = "qwen-max", api_key = os.getenv("LLM_API_KEY"))

# 调用invoke向模型提问
res = model.invoke(input="你是谁？")
print(type(res))                        # str
print(res)

# 调用 stream 方法向模型提问 - 流式输出
res = model.stream(input="什么是人工智能？")
for chunk in res:                       # type(res) : <class 'generator'>
    # print(type(chunk))                # str
    print(chunk, end="", flush=True)
```

#### (2) Chat Model
Chat Model 是目前 LangChain 中最常用的模型类型。
输入是Langchain的**消息类**，一共有三类分别为 `SystemMessage`、`AIMessage`和 `HumanMessage`。从 ` langchain_core.messages`中导入消息类。

**注**：如果采用Langchain框架中三方sdk如 `langchain_community.chat_models.tongyi.ChatTongyi`,后期切换api不便，所以采用大多三方api兼容的openai sdk。

参考示例如下：采用**openai sdk**调用chat model

- 导入依赖

```
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
```

- 创建客户端

```
model = ChatOpenAI(
    model = os.getenv("LLM_MODEL_ID"),
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL")
)

messages = [
    SystemMessage(content="你是我的人工智能助手，协助我解答问题。"),
    HumanMessage(content="请介绍一下自己。")
]
```

- 调用 invoke - 直接输出

```
# response = model.invoke(messages)
# print(type(response))               # <class 'langchain_core.messages.ai.AIMessage'>
# print(response)                     # 调用继承于BaseMessage的__str__方法，输出消息内容

# AIMessage → BaseMessage → Serializable → BaseModel
```

- 调用 stream - 流式输出

```
response_stream = model.stream(messages)
print(type(response_stream))        # <class 'generator'>
for chunk in response_stream:
    print(type(chunk))              # <class 'langchain_core.messages.ai.AIMessageChunk'>
    print(chunk.content,end="",flush=True)                     # 每次迭代输出一个消息块，直到流结束

# AIMessageChunk → AIMessage，BaseMessageChunk → BaseMessage → Serializable → BaseModel
```

**补充**：
对于messages可以采用如下方式进行简写：
```
messages = [
    SystemMessage(content="你是我的人工智能助手，协助我解答问题。"),
    HumanMessage(content="请介绍一下自己。")
]

# 简写
messages = [
    {"role": "system", "content": "你是我的人工智能助手，协助我解答问题。"},
    {"role": "user", "content": "请介绍一下自己。"}
]
```

#### (3) Embedding Model
Embedding Model 不负责聊天，它负责把文本变成向量。
下述例子调用阿里云的Embedding Model
```
# 导入依赖
from langchain_commnity.embeddings import DashScopeEmbeddings

# 定义模型
embed = DashScopeEmbeddings(
    model = os.getenv("EMBEDDING_MODEL_ID"), 
    dashscope_api_key = os.getenv("EMBEDDING_API_KEY"),
)

# 将字符串转换为向量 - embed_query() | embed_documents()
print(embed.embed_query("hello world")) # 单个字符串转换
print(embed.embed_documents(["hello world", "你好，世界"])) # 批量转换

```
