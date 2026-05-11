from langchain_community.document_loaders import JSONLoader
import os,json

current_dir = os.path.dirname(os.path.abspath(__file__))

loader = JSONLoader(
    # 必要参数
    file_path=os.path.join(current_dir, "data/json_loader_json.json"),
    jq_schema= ".",         # 选择整个 JSON 文件作为一个文档
    # 可选参数
    text_content=False,     # 指明抽取部分是不是str类型，默认为True，如果抽取部分不是str类型，设置为False, 会自动将抽取部分转换为str类型，再传入Document的page_content中
    json_lines = False      # JSON 文件是否为 JSON Lines 格式
)

docs = loader.load()
for doc in docs:
    # page_content的保存类型是字符串,不存储中文，会被转义成unicode编码
    # print(type(doc.page_content))
    # print(doc.page_content)

    # 先将包含unicode编码的字符串转换回Python对象（如字典），再使用json.dumps()将其转换为格式化的JSON字符串，这样就可以正确显示中文了。
    doc_content_dict = json.loads(doc.page_content)
    json_obj = json.dumps(doc_content_dict, ensure_ascii=False, indent=4)
    print(type(json_obj))
    print(json_obj)
  
