from langchain_community.document_loaders import CSVLoader
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

loader = CSVLoader(
    file_path = os.path.join(current_dir, "data/data.csv"),
    encoding="utf-8",           # CSV文件的编码格式，默认为utf-8 
    csv_args={
        "delimiter": ",",       # CSV文件的分隔符，默认为逗号
        "quotechar": '"',      # CSV文件中用于引用字段的字符，默认为双引号
        # "fieldnames": ["name", "age", "gender"],  # 缺失时字段名时指定。
    }
)

documents = loader.load()   # documents: List[Document]
for doc in documents:
    print(doc)


documents = loader.lazy_load()   # documents: Iterable[Document]
for doc in documents:
    print(doc)

