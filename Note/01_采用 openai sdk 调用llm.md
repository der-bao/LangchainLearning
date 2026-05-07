## 1. 采用 openai sdk 调用llm

先采用`client = OpenAI(api_key="...",base_url="...")`创建一个客户端，然后采用如下方法获取回复。
```
response = client.chat.completions.create(
    model = "...",
    messages = [
        # 一共有三种角色,分别为 system/assistant/user
        {"role": "system", "content": "你是我的人工智能助手，帮助我解答问题。"},
        {"role": "user", "content": "你是谁？"},
        ...
    ],
    stream = True # optional, 开启后流式输出回复
)
```

- 当**直接输出**时，采用`print(response.choices[0].message.content)`得到存储的内容。
此时，response是一个ChatCompletion对象，具体的结构如下展示：
 
```
ChatCompletion(
    id='chatcmpl-f714fc04-be92-9935-834b-a8dffa91bef0',
    choices=[
        Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
                content='我是你的AI助手，旨在帮助你解答问题、提供信息或协助完成各种任务。有什么我可以帮你的吗？😊',
                refusal=None,
                role='assistant',
                annotations=None,
                audio=None,
                function_call=None,
                tool_calls=None
            )
        )
    ],
    created=1778124156,
    model='qwen3-max',
    object='chat.completion',
    service_tier=None,
    system_fingerprint=None,
    usage=CompletionUsage(
        completion_tokens=26,
        prompt_tokens=26,
        total_tokens=52,
        completion_tokens_details=None,
        prompt_tokens_details=PromptTokensDetails(
            audio_tokens=None,
            cached_tokens=0
        )
    )
)
```

| 字段 | 含义 |
|---|---|
| `id` | 本次对话请求的唯一 ID |
| `choices` | 模型生成的回复结果列表 |
| `finish_reason='stop'` | 模型正常结束生成 |
| `message.content` | AI 实际回复内容 |
| `role='assistant'` | 当前消息角色 |
| `created` | 创建时间（Unix 时间戳） |
| `model='qwen3-max'` | 使用的大模型名称 |
| `object='chat.completion'` | 返回对象类型 |
| `usage` | token 使用统计 |

**因此**，实际的输出内容存储在`choices[0].message.content`


- **流式输出**
当采用流式输出时，response是一个**openai.Stream**对象，可以视为一个生成器，不断yield出**ChatCompletionChunk**对象,采用如下方式实现流式输出。

```
response = client.chat.completions.create(
    model= os.getenv("LLM_MODEL_ID", "qwen3-max"),
    messages=[
        {"role": "system", "content": "你是我的人工智能助手，帮助我解答问题。"},
        {"role": "user", "content": "什么是人工智能？"}
    ],
    stream = True
)

for chunk in response:
    if chunk.type == "response.output_text.delta":
        print(chunk.choices[0].delta.content, end="", flush=True)
```

其中，每个**ChatCompletionChunk**对象的具体结构如下：

```
ChatCompletionChunk(
    id='chatcmpl-24249867-1daf-9f7b-8be7-5766e8aa3c48',
    choices=[
        Choice(
            delta=ChoiceDelta(
                content='人工智能',
                role=None
            ),
            finish_reason=None,
            index=0
        )
    ],
    created=1778125253,
    model='qwen3-max',
    object='chat.completion.chunk'
)
```

**因此**，对于每一个切片通过`chunk.choices[0].delta.content`获取实际存储的内容。







