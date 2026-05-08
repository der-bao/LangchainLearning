from langchain_core.prompts import ChatPromptTemplate

# 创建一个聊天提示词模板
chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是我的人工智能助手，协助我解答问题。"),
    ("user", "请介绍一下这个概念{concept}。")
])
print(type(chat_prompt_template))    # <class 'langchain_core.prompts.chat.ChatPromptTemplate'>

# 注入模板
prompt = chat_prompt_template.invoke(input={"concept": "Langchain"})
print(type(prompt))            # <class 'langchain_core.prompts.chat.ChatPromptValue'>
print(prompt)

"""
messages=[
SystemMessage(content='你是我的人工智能助手，协助我解答问题。', additional_kwargs={}, response_metadata={}), 
HumanMessage(content='请介绍一下这个概念Langchain。', additional_kwargs={}, response_metadata={})
]
"""
print(prompt.to_string())
