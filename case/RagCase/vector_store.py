from langchain_chroma import Chroma
import config_data as config


class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name = config.collection_name,
            embedding_function = self.embedding,
            persist_directory = config.persist_directory
        )

    def get_retriever(self):
        """返回向量检索器，方便加入链"""
        return self.vector_store.as_retriever(search_kwargs={"k": config.similarity_threshold})
    
# test
if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    vector_service = VectorStoreService(DashScopeEmbeddings(model = config.embedding_model_name))
    retriever = vector_service.get_retriever()
    query = "我身高170cm，尺码推荐"
    result = retriever.invoke(query)
    print(result)