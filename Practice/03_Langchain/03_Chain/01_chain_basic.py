from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("LLM_MODEL_ID"),
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL")
)

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是我的人工智能助手，协助我解答问题。"),
    MessagesPlaceholder("history"),
    ("user", "{question}")
])

history = [
    {"role": "user", "content": "今天周几？"},
    {"role": "ai", "content": "今天周五。"}        
]


question = "明天周几？"

chain = chat_prompt_template | model
print(type(chain))                          # <class 'langchain_core.runnables.base.RunnableSequence'>

res = chain.stream(input={
    "history": history,
    "question": question
})

for chunk in res:
    if chunk.content:
        print(chunk.content, end="",flush=True)








