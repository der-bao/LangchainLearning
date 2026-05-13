import os

# 请在这里配置你的阿里云 DashScope API KEY
os.environ["DASHSCOPE_API_KEY"] = "sk-82253b0148f04b589c8d3105c79e76ac"
dashscope_api_key = os.environ["DASHSCOPE_API_KEY"]

md5_path = "./md5.txt"

# chroma
collection_name = "mycollection"
persist_directory = "./chroma_db"
similarity_threshold = 3            # 检索返回匹配的文档样本


# spliter
chunk_size = 1000
chunk_overlap = 100
separators = ["\n\n", "\n", ".", "!", "?", "。", "！", "？", " ", ""]
max_spliter_char_number = 1000  # 当文本长度超过这个值时才进行分割，否则不分割，直接存储到向量库中

# model 
embedding_model_name = "text-embedding-v4"
chat_model_name = "qwen3-max"

# custom
session_config = {
        "configurable": {
            "session_id": "user_001"
        }   
    }