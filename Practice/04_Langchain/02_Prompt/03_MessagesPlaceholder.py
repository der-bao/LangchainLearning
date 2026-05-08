# 导入依赖
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 创建模板
chat_prompt_template = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "你是我的人工智能助手，协助我解答问题。"},
    MessagesPlaceholder("history"),
    {"role": "user", "content": "{question}"}
])

# 示例历史信息
history = [
    {"role": "user", "content": "今天周几？"},
    {"role": "ai", "content": "今天周五。"}        
]

question = "明天周几？"


# 注入模板
prompt = chat_prompt_template.invoke(input={"history":history, "question":question})
print(type(prompt))         # <class 'langchain_core.prompt_values.ChatPromptValue'>
print(prompt.to_string())   # 打印注入后的字符串

"""
    System: 你是我的人工智能助手，协助我解答问题。
    Human: 今天周几？
    AI: 今天周五。
    Human: 明天周几？
"""