# Embedding Model
# 将输入的字符串，转化为一个向量

# ================ 调用阿里云的Embedding Model =====================
from dotenv import load_dotenv
import os
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

# 初始化嵌入模型对象，默认使用模型是 text-embedding-v1
embed = DashScopeEmbeddings(
    model = os.getenv("EMBEDDING_MODEL_ID"), 
    dashscope_api_key = os.getenv("EMBEDDING_API_KEY"),
)

# 测试
print(embed.embed_query("hello world")) # 单个字符串转换
print(embed.embed_documents(["hello world", "你好，世界"])) # 批量转换


# # ================ 调用Ollama的Embedding Model =====================
# from langchain_ollama import OllamaEmbeddings

# # 初始化嵌入模型对象
# emded = OllamaEmbeddings(model="qwen3-embedding")

# print(emded.embed_query("hello world")) # 单个字符串转换
# print(emded.embed_documents(["hello world", "你好，世界"])) # 批量转换