# 采用Openai SDK调用LLM

"""
相关知识：
    - SDK(Software Development Kit)是一种软件开发工具包，提供了一组预先编写的代码库、工具和文档，帮助开发者更轻松地使用特定平台或服务的功能。
    - OpenAI SDK是OpenAI提供的软件开发工具包，允许开发者通过编程方式访问和使用OpenAI的语言模型（如GPT-3、GPT-4等）来构建各种应用程序。
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # 从 .env 文件加载环境变量

api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model_id = os.getenv("LLM_MODEL_ID")

# ========= 不采用SDK，直接调用API接口 =========
# import requests

# url = f"{base_url}/chat/completions"
# headers = {
#     "Authorization": f"Bearer {api_key}",
#     "Content-Type": "application/json"  
# }
# data = {
#     "model": model_id,      
#     "messages": [
#         {"role": "system", "content": "你是我的人工智能助手，帮助我解答问题。"},
#         {"role": "user", "content": "什么是人工智能？"}
#     ]   
# }   

# response = requests.post(url, headers=headers, json=data)
# if response.status_code == 200:
#     result = response.json()
#     print(result.choices[0].message.content)


# ========= 采用 openai SDK调用LLM =========

client = OpenAI(
    api_key = os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)

# # 老接口
# response = client.chat.completions.create(
#     model= os.getenv("LLM_MODEL_ID", "qwen3-max"),
#     messages=[
#         {"role": "system", "content": "你是我的人工智能助手，帮助我解答问题。"},
#         {"role": "user", "content": "你是谁？"}
#     ]
# )

# print(response)
# print(response.choices[0].message.content)

response = client.chat.completions.create(
    model= os.getenv("LLM_MODEL_ID", "qwen3-max"),
    messages=[
        {"role": "system", "content": "你是我的人工智能助手，帮助我解答问题。"},
        {"role": "user", "content": "什么是人工智能？"}
    ],
    stream = True
)

print(response)

for chunk in response:
    print(chunk)
    # print(chunk.choices[0].delta.content, end="", flush=True)











# 新接口
# respose = client.responses.create(
#     model = os.getenv("LLM_MODEL_ID", "qwen3-max"),
#     input = [
#         {"role": "system", "content": "你是我的人工智能助手，帮助我解答问题。"},
#         {"role": "user", "content": "什么是人工智能？"}
#     ]
# )
# print(respose.output_text)

# 新接口流式输出
# response = client.responses.create(
#     model = os.getenv("LLM_MODEL_ID", "qwen3-max"),
#     input = [
#         {"role": "system", "content": "你是我的人工智能助手，帮助我解答问题。"},
#         {"role": "user", "content": "什么是人工智能？"}
#     ],
#     stream = True
# )

# for chunk in response:
#     if chunk.type == "response.output_text.delta":
#         print(chunk.delta, end="", flush=True)
