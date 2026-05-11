# 导入依赖
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

loader = TextLoader(
    os.path.join(current_dir, "data/text_loader.txt"), 
    encoding="utf-8"
)

docs = loader.load()

print(len(docs))  # 输出加载的文档数量

# 定义文本分割器
txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,         # 每个文本块的最大长度 
    chunk_overlap=50,       # 文本块之间的重叠长度
    separators=["\n\n", "\n", " ", ""],     # 分割文本的的符号以及优先级顺序
    length_function=len,    # 计算文本长度的函数
)

split_docs = txt_splitter.split_documents(docs)
print(len(split_docs))  # 输出分割后的文本块数量

for i, doc in enumerate(split_docs):
    print(f"========== 文本块 {i + 1} ==========")
    print(type(doc))    # 输出文本块内容的类型
    print(doc.page_content)
    print("======================================")
