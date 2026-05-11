from langchain_community.document_loaders import PyPDFLoader
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

loader = PyPDFLoader(
    file_path=os.path.join(current_dir, "data/智能汽车创新发展战略.pdf"),
    mode='page',          # 读取模式：'page': 将每一页作为一个Document对象; 'single': 将整个PDF作为一个Document对象
    password=None,        # PDF文件的密码，如果有的话
)

docs = loader.load()
for count, doc in enumerate(docs):
    print(f"Document {count + 1}: {doc.page_content}")