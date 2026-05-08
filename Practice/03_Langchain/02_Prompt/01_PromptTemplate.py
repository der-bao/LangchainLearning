from langchain_core.prompts import PromptTemplate

# 创建模板
prompt_template = PromptTemplate.from_template(
    "今天是{weekday}，心情不错。"
)

print(type(prompt_template))            # <class 'langchain_core.prompts.prompt.PromptTemplate'>
print(prompt_template)

# 注入模板
# prompt = prompt_template.invoke(input={"weekday":"周五"})   # <class 'langchain_core.prompt_values.StringPromptValue'>
prompt = prompt_template.format(weekday="周五")               # <class 'str'>，直接返回字符串

print(type(prompt))
print(prompt)


