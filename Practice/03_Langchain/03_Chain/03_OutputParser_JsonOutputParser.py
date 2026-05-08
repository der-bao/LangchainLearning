from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatOpenAI(
    model = os.getenv("LLM_MODEL_ID"),
    api_key = os.getenv("LLM_API_KEY"),
    base_url = os.getenv("LLM_BASE_URL")
)

first_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "我的邻居姓:{lastname},刚生了{gender},请起名，并封装到JSON格式返回给我。要求key是name，value是名字，请严格遵守格式要求。")
])

second_prompt_template = ChatPromptTemplate.from_messages([
     ("user", "姓名{name},请帮我解析含义")
]) 


chain = first_prompt_template | model | JsonOutputParser() | second_prompt_template | model
res = chain.invoke(input={"lastname": "张", "gender": "男"})
print(res.content)