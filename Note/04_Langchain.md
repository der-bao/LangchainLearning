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
# print(response.content)                     # 调用继承于BaseMessage的__str__方法，输出消息内容

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

### 2. Prompt

在 LangChain 中，Prompt 组件负责把你的输入变量组织成模型能理解的提示词。
![1778212086309](image/04_Langchain/1778212086309.png)

```
# Langchain的Prompt组件
- 主要学习了三个类PromptTemplate、Chat Prompt Template和MessagesPlaceholder
- PromptTemplate 适合**文本补全模型**或者简单字符串提示词；ChatPromptTemplate适合聊天模型
- PromptTemplate采用from_template创建模板；ChatPromptTemplate采用from_messages创建模板
- PromptTemplate和Chat Prompt Template都具有两种模板注入方式：invoke()和format()。前者得到的是PromptValue系列的对象（采用to_string()得到字符串），后者直接得到字符串。
- 在创建ChatPromptTemplate时，添加MessagesPlaceholder可以添加history变量，达到注入历史的功能。
```

#### (1) PromptTemplate：普通文本提示词模板

PromptTemplate 适合**文本补全模型**或者简单字符串提示词。

```
# 导入依赖
from langchain_core.prompts import PromptTemplate

# 创建模板 - 调用 **PromptTemplate.from_template()** 方法
prompt_template = PromptTemplate.from_template(
    "今天是{weekday}，心情不错。"
)

# 注入模板
# (1) 调用底层的invoke(), 输入是 "input={"插入键":"插入值",...}"
prompt = prompt_template.invoke(input={"weekday":"周五"})   
print(type(prompt))     # <class 'langchain_core.prompt_values.StringPromptValue'>
print(prompt)           # text='今天是周五，心情不错。'

# (2) 调用fromat() 
prompt = prompt_template.format(weekday="周五")         
print(type(prompt))     # <class 'str'>，直接返回字符串
print(prompt)           # 今天是周五，心情不错。

```

#### (2) ChatPromptTemplate

```
# 导入依赖
from langchain_core.prompts import ChatPromptTemplate

# 创建模板 - 调用ChatPromptTemplate.from_messages()
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是我的人工智能助手，协助我解答问题。"),
    ("user", "请介绍一下这个概念{concept}。")
])
print(type(chat_prompt_template))    # <class 'langchain_core.prompts.chat.ChatPromptTemplate'>

# 注入模板 - 调用底层的invoke()
prompt = chat_prompt_template.invoke(input={"concept": "Langchain"})
print(type(prompt))                  # <class 'langchain_core.prompts.chat.ChatPromptValue'>
print(prompt)

"""
    messages=[
    SystemMessage(content='你是我的人工智能助手，协助我解答问题。', additional_kwargs={}, response_metadata={}), 
    HumanMessage(content='请介绍一下这个概念Langchain。', additional_kwargs={}, response_metadata={})]
"""

# 如果想要直接输出字符串，调用to_string()方法
print(prompt.to_string())

"""
    System: 你是我的人工智能助手，协助我解答问题。
    Human: 请介绍一下这个概念Langchain。
"""
```

#### (3) MessagesPlaceholder

如果想要创建一个可以加入历史对话信息的ChatPromptTemplate，可以引入消息占位符MessagesPlaceholder。

```
# 导入依赖
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建模板
chat_prompt_template = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "你是我的人工智能助手，协助我解答问题。"},
    MessagesPlaceholder("history"),
    {"role": "user", "content": "{question}"}
])

# 示例历史信息
history = [
    {"role": "user", "content": "今天周几？"}，
    {"role": "assistant", "content": "今天周五。"}      # 这里的角色也可以是"ai"
]

question = "明天周几？"


# 注入模板
prompt = chat_prompt_template.invoke(input={"history":history, "question":question})
print(type(prompt))         # # <class 'langchain_core.prompt_values.ChatPromptValue'>
print(prompt)

"""
    System: 你是我的人工智能助手，协助我解答问题。
    Human: 今天周几？
    AI: 今天周五。
    Human: 明天周几？
"""
```
