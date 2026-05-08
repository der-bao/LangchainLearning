from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("LLM_MODEL_ID"),  
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL")
)

# 记忆内存存储
store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

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

config = {"configurable": {"session_id": "test_session"}}

for count, q in enumerate(question):
    res = conversation_chain.invoke(input={"input": q}, config=config)
    print(f"========== 第 {count + 1} 个问题的模型回复 ==========")
    print(res.content)
    print("======================================")

