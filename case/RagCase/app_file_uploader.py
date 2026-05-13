"""
基于Streamlit框架完成web网页上传服务

项目主文件
"""

import streamlit as st
import time
from knowledge_base import KnowledgeBaseService


if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()


# 添加网页标题
st.title("文件上传服务")

# 创建文件上传组件
file_uploader = st.file_uploader(
    "请选择要上传的文件", 
    type=["txt"], 
    accept_multiple_files=False         # 只允许上传单个txt文件
    )

if file_uploader is not None:
    # 读取上传的文件基本信息
    file_name = file_uploader.name
    file_type = file_uploader.type
    file_size = file_uploader.size / 1024  # 转换为KB

    # 基础信息
    st.subheader(f"文件名：{file_name}")    # 二级标题
    st.write(f"文件类型：{file_type} | 文件大小：{file_size:.2f} KB")

    # get_value -> bytes数组 -> decode('utf-8')
    file_content = file_uploader.getvalue()     # b'streamlit\xe6\x96\x87\xe4\xbb\xb6\xe4\xb8\x8a\xe4\xbc\xa0\r\n'
    file_content = file_content.decode('utf-8') # 编码为字符串
    print(file_content)
    
    with st.spinner("正在上传文件..."):
        time.sleep(1)   # 模拟上传过程中的等待时间
        result = st.session_state["service"].upload_by_str(file_content, file_name)
        st.write(result)









    
