from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("LLM_MODEL_ID"),
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL")
)

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

input = "春眠不觉晓的下一句是什么？"

def print_prompt(prompt_value):
    print("========== 当前生成的提示词 ==========")
    print(prompt_value.to_string())
    print("======================================")
    return prompt_value


chain = chat_prompt_template | RunnableLambda(print_prompt) | model | StrOutputParser() 
res = chain.invoke(input={"input": input})
print(res)