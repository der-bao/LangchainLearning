from dotenv import load_dotenv
import os 

from langchain_community.embeddings import DashScopeEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import VectorParams, Distance


load_dotenv()

embed = DashScopeEmbeddings(
    model = os.getenv("EMBEDDING_MODEL_ID"), 
    dashscope_api_key = os.getenv("EMBEDDING_API_KEY"),
)

client = QdrantClient(
    host = os.getenv("QDRANT_HOST"),
    port = os.getenv("QDRANT_PORT")
)

# 初始化一个空的 QdrantVectorStore
collection_name = "test_collection"

if not client.collection_exists(collection_name):
    print(f"集合 {collection_name} 不存在，正在创建...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE) # 阿里云 text-embedding-v1/v2 是 1536 维
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embed
)

print(f"Vector Store 初始化成功: {vector_store}")

# ========================================
# 插入向量

from langchain_core.documents import Document

docs = [
    Document(page_content="Qdrant 是高性能向量数据库", metadata={"source": "wiki"}),
    Document(page_content="LangChain 是 LLM 应用框架", metadata={"source": "docs"})
]

# 写入 Qdrant（自动生成向量 → 变成 Point 存入）
vector_store.add_documents(docs)


# ========================================
# 查询向量

results = vector_store.similarity_search("向量数据库", k=1)     # list[Document]

for doc in results:
    print(doc.page_content)
    print(doc.metadata)

# ========================================
# 结构化过滤查询
# 只查 source=wiki 的文档
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = vector_store.similarity_search(
    "向量数据库",
    k=1,
    filter=Filter(
        must=[
            FieldCondition(
                key="metadata.source", # 在 Langchain 中，Document 的 metadata 会被嵌套存在 payload 的 "metadata" 键下
                match=MatchValue(value="wiki")
            )
        ]
    )
)
print("结构化过滤后查询结果：")
for doc in results:
    print(doc.page_content)
    print(doc.metadata)