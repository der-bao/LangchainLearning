# 1. 安装(Linux)
```
curl -fsSL https://ollama.com/install.sh | sh
ollama --version    # 查看版本
```

# 2. 开启服务(默认绑定在本地的11434端口)
```
ollama serve
```

# 3. 下载模型(一般没有必要,运行时会自动下载)
```
ollama pull qwen3-max
```

# 4. 交互式使用
```
ollama run qwen2.5:7b
```

# 5. 通过 OpenAI SDK 使用 Ollama
```
client = OpenAI(
    api_key="ollama",          # Ollama 不需要真实的 API Key，但 OpenAI SDK 要求此字段不能为空
    base_url="http://localhost:11434/v1", # Ollama 提供的 OpenAI 兼容接口地址
)
```

# 其他指令
```
ollama list    # 列出已安装的模型
```

