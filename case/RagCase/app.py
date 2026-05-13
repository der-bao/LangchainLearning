import streamlit as st
import time
from rag import RagService
from knowledge_base import KnowledgeBaseService
import config_data as config

# 侧边栏：知识库管理
with st.sidebar:
    st.subheader("📁 知识库管理")
    file_uploader = st.file_uploader(
        "请选择要上传的参考文件", 
        type=["txt"], 
        accept_multiple_files=False
    )
    
    if "service" not in st.session_state:
        st.session_state["service"] = KnowledgeBaseService()
        
    if file_uploader is not None:
        file_name = file_uploader.name
        file_size = file_uploader.size / 1024  # 转换为KB
        
        st.write(f"文件大小：{file_size:.2f} KB")
        file_content = file_uploader.getvalue().decode('utf-8')
        
        with st.spinner("正在处理并向量化文件..."):
            time.sleep(1)
            result = st.session_state["service"].upload_by_str(file_content, file_name)
            st.success(result)

# title
st.title("智能客服")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "您好！我是您的智能客服，请问有什么可以帮助您的？"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 打印历史消息
for message in st.session_state["message"]:
    st.chat_message(message["role"]).markdown(message["content"])   


# 在页面底部显示用户输入框
prompt = st.chat_input("请输入您的问题：")


if prompt:
    # 显示用户输入的消息
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 模拟客服回复
    ai_res_list = []
    with st.spinner("智能客服正在思考..."):
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config.session_config)

        def capture(generator,cache_list):
            for chunk in generator:
                cache_list.append(chunk)    # 将生成的内容保存到cache_list中
                yield chunk

        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))  # 将生成的内容流式输出到页面上
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})  # 将生成的内容保存到历史消息中，供后续使用

