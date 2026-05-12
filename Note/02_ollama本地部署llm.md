# 1. 安装

Linux：
```
curl -fsSL https://ollama.com/install.sh | sh
ollama --version    # 查看版本
```

Windows / macOS：可以直接去 Ollama 官网下载安装包。安装后确认命令可用：

```
ollama --version
```

# 2. 开启服务(默认绑定在本地的11434端口)
```
ollama serve
```

# 3. 下载模型

可以先手动下载，也可以在 `ollama run` 时自动下载。注意模型名要使用 Ollama 支持的名称，例如：
```
ollama pull qwen2.5:7b
ollama pull deepseek-r1:1.5b
```

# 4. 交互式使用
```
ollama run qwen2.5:7b
```

# 5. 通过 OpenAI SDK 使用 Ollama
```
from openai import OpenAI

client = OpenAI(
    api_key="ollama",          # Ollama 不需要真实的 API Key，但 OpenAI SDK 要求此字段不能为空
    base_url="http://localhost:11434/v1", # Ollama 提供的 OpenAI 兼容接口地址
)

response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": "你是我的人工智能助手。"},
        {"role": "user", "content": "请介绍一下自己。"}
    ]
)

print(response.choices[0].message.content)
```

# 其他指令
```
ollama list    # 列出已安装的模型
ollama ps      # 查看正在运行的模型
ollama rm 模型名 # 删除本地模型
```

