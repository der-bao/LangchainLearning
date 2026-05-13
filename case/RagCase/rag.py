from dotenv import load_dotenv
import os

import config_data as config
from vector_store import VectorStoreService
from file_history_store import get_history

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI

from langchain_core.runnables import RunnablePassthrough 
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


load_dotenv()  # 加载环境变量

class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(DashScopeEmbeddings(model = config.embedding_model_name, dashscope_api_key = os.getenv("EMBEDDING_API_KEY"),))

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料：{context}"),
                ("system", "并且我提供用户的对话历史记录，如下："),
                MessagesPlaceholder("chat_history"),
                ("user", "用户提问：{input}")
            ]
        )

        self.chat_model = ChatOpenAI(
            model = config.chat_model_name,
            api_key = os.getenv("LLM_API_KEY"),
            base_url = os.getenv("LLM_BASE_URL")
        )
        
        self.chain = self.__get_chain()

    def __get_chain(self):
        " 获取最终的执行链"
        retriever = self.vector_service.get_retriever()

        def format_func(retrieved_docs):
            if not retrieved_docs:
                return "无相关参考资料"
            reference_docs = []
            for doc in retrieved_docs:
                reference_docs.append(doc.page_content)
            return reference_docs

        def print_prompt(full_prompt):
            print("=== 完整提示词如下 ===")
            print(full_prompt.to_string())
            print("=====================")
            return full_prompt

        def format_for_retriever(input_dict):
            """
            retriever
            - 输入：用户输入的文本 str

            但是RunnableWithMessageHistory传入retriever的输入是一个字典，{"input": input_text, "chat_history": []

            因此要将输入字典中的用户输入提取出来，传给retriever

            dict -> str
        
            """
            return input_dict["input"]  # 从输入字典中提取用户输入的文本

        def format_for_prompt(input_dict):
            """
            第一个组件的输出是：
            {
                "input": {"input" : input_text, "chat_history": []},
                "context": ["相关文档1内容", "相关文档2内容", ...] 
            }

            但是prompt模板需要的输入是：
            {
                "input": input_text,
                "context": ["相关文档1内容", "相关文档2内容", ...],
                "chat_history": []
            }

            """
            return {
                "input": input_dict["input"]["input"],  # 从输入字典中提取用户输入的文本
                "context": input_dict["context"],        # 相关文档内容
                "chat_history": input_dict["input"]["chat_history"]  # 对话历史记录
            }

        chain = (
            {
                "input": RunnablePassthrough(), 
                "context": RunnableLambda(format_for_retriever) | retriever | format_func
            } | RunnableLambda(format_for_prompt) | self.prompt_template | print_prompt  | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,                            # 通过会话id获取InMemoryChatMessageHistory对象的函数
            input_messages_key = "input",           # 表示用户输入在模板中的占位符
            history_messages_key = "chat_history"    # 表示历史消息在模板中的占位符
        )

        return conversation_chain

# test 
if __name__ == "__main__":
    input_text = "针织毛衣怎么保养？"  
    
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }   
    }

    answer = RagService().chain.invoke({"input": input_text}, session_config)

    print("=== 最终回答如下 ===")
    print(answer)
    print("=====================")
    