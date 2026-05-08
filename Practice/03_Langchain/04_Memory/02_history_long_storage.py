from dotenv import load_dotenv
import os
from typing import List
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory  # 需要继承的基类
from langchain_core.messages import message_to_dict, messages_from_dict

class FileChatMessageHistory(BaseChatMessageHistory):
    """基于文件存储的聊天记录"""
    def __init__(self, session_id:str, storage_path:str):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(storage_path, f"{session_id}.json")

    @property
    def messages(self) -> List[BaseChatMessageHistory]:
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r", encoding="utf-8") as f:
            messages_dict = json.load(f)
        return messages_from_dict(messages_dict)

    def add_message(self, message):
        messages = self.messages
        messages.append(message)
        messages_dict = [message_to_dict(m) for m in messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(messages_dict, f, ensure_ascii=False, indent=4)   # indent代表缩进格式化输出

    def clear(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


load_dotenv()

model = ChatOpenAI(
    model = os.getenv("LLM_MODEL_ID"),
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL")
)

def get_session_history(session_id):
    current_dir = os.path.dirname(__file__)
    storage_path = os.path.join(current_dir, "chat_history")
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    return FileChatMessageHistory(session_id, storage_path)

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."), 
    MessagesPlaceholder(variable_name="history"),  
    ("user", "{input}")
])

question = [
    "春眠不觉晓的下一句是什么？",
    "夜来风雨声的下一句是什么？",
]

def print_prompt(prompt_value):
    print("========== 当前生成的提示词 ==========")
    print(prompt_value.to_string())
    print("======================================")
    return prompt_value

chain = chat_prompt_template | print_prompt | model

conversation_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "user01_001"}}

for count, q in enumerate(question):
    res = conversation_chain.invoke(input={"input": q}, config=config)
    print(f"========== 第 {count + 1} 个问题的模型回复 ==========")
    print(res.content)
    print("======================================")

