from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient

load_dotenv()

# 1. 定义客户端
client = QdrantClient(
    host = os.getenv("QDRANT_HOST"),
    port = os.getenv("QDRANT_PORT")
)

print(client.get_collections())


# 2. 创建一个新的 collection
from qdrant_client.models import Distance, VectorParams

collection_name = os.getenv("QDRANT_COLLECTION") # 从 QDRANT_COLLECTION 获取
vector_size = int(os.getenv("QDRANT_VECTOR_SIZE"))

# 如果集合不存在则创建
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,               
            distance=Distance.COSINE
        )
    )

print(client.get_collections())

# 3. 插入向量

from qdrant_client.models import PointStruct

# 补齐维度以匹配设置的 384 维度
test_vector = [0.1, 0.2, 0.3, 0.4] + [0.0] * (vector_size - 4)

client.upsert(
    collection_name=collection_name,
    points=[                            # List[PointStruct]
        PointStruct(
            id=1,
            vector=test_vector,
            payload={
                "text": "JWT 是 Token 格式",
                "category": "backend"
            }
        ),
        PointStruct(
            id=2,
            vector=test_vector,
            payload={
                "text": "text2",
                "category": "backend"
            }
        ),
        PointStruct(
            id=3,
            vector=[0.15]*vector_size,
            payload={
                "text": "text3",
                "category": "frontend"
            }
        )
    ]
)

print("插入成功")


# 5. 查询向量
query_vector = [0.15]*vector_size  # 查询向量，维度与插入的向量相同

# 搜索 top-2 最相似
search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=2
)

print(type(search_result))          # <class 'qdrant_client.http.models.models.QueryResponse'>

# 打印结果
for hit in search_result.points:
    print(f"ID: {hit.id}, 得分: {hit.score:.4f}, 内容: {hit.payload}")


# 先结构化过滤再查询
from qdrant_client.models import Filter, FieldCondition, MatchValue

# 只搜 category = "backend" 的结果
search_result = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="backend")
            )
        ]
    ),
    limit=2
)

print("结构化过滤后查询结果：")
for hit in search_result.points:
    print(f"ID: {hit.id}, 得分: {hit.score:.4f}, 内容: {hit.payload}")