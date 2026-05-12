## 1. 采用 OpenAI SDK 调用 LLM

先采用 `client = OpenAI(api_key="...", base_url="...")` 创建一个客户端，然后调用模型接口获取回复。这里的 `base_url` 可以是 OpenAI 官方地址，也可以是阿里云百炼、Ollama 等兼容 OpenAI API 的服务地址。
```
response = client.chat.completions.create(
    model = "...",
    messages = [
        # 常用角色：system / user / assistant
        {"role": "system", "content": "你是我的人工智能助手，帮助我解答问题。"},
        {"role": "user", "content": "你是谁？"},
        ...
    ],
    stream = True # optional, 开启后流式输出回复
)
```

- 当**非流式输出**时，采用 `print(response.choices[0].message.content)` 得到回复内容。
此时，`response` 是一个 `ChatCompletion` 对象，具体结构如下：
 
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

**因此**，实际的输出内容存储在 `choices[0].message.content`。


- **流式输出**
当采用 Chat Completions 的流式输出时，`response` 是一个 `openai.Stream` 对象，可以视为一个迭代器，不断 yield 出 `ChatCompletionChunk` 对象，采用如下方式实现流式输出。

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
    if chunk.choices[0].delta.content:
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

**因此**，对于每一个切片通过 `chunk.choices[0].delta.content` 获取实际增量内容。

注意：`chunk.type == "response.output_text.delta"` 和 `chunk.delta` 是 **Responses API** 流式输出中的写法，不适用于上面的 `client.chat.completions.create(..., stream=True)`。如果使用 Responses API，写法类似：

```
response = client.responses.create(
    model=os.getenv("LLM_MODEL_ID"),
    input=[
        {"role": "system", "content": "你是我的人工智能助手，帮助我解答问题。"},
        {"role": "user", "content": "什么是人工智能？"}
    ],
    stream=True
)

for chunk in response:
    if chunk.type == "response.output_text.delta":
        print(chunk.delta, end="", flush=True)
```







