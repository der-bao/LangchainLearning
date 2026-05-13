"""
    知识库

    checj_md5()：检查知识库里有没有重复的文件
    save_md5()：将每一份文件的md5值存储在md
    get_string_md5()：获取字符串的md5值

    KnowledgeBaseService类
    - __init__()：初始化函数，创建Chroma向量库对象和文本分割器对象
    - upload_by_str(content, filename): 将传入的字符串进行向量化，存入向量数据库中
"""

import os
import config_data as config
import hashlib
from datetime import datetime

from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter


def check_md5(md5_str):
    """"
        检查知识库里有没有重复的文件
        对于每一份文件都会有一个特有的md5值存储在md5.txt里，如果有重复的文件就会有重复的md5值
    """
    
    current_dir = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件所在的目录
    md5_path = os.path.join(current_dir, config.md5_path)    # 拼接出md5.txt文件的完整路径

    if not os.path.exists(md5_path):
        with open(md5_path,"w",encoding="utf-8") as f:
            f.write("")    # 创建一个空的md5.txt文件
        return False
    else:
        with open(md5_path,"r",encoding="utf-8") as f:
            for line in f.readlines():
                if line.strip() == md5_str:     # strip()去掉字符串首尾的空格和换行符 
                    return True
            return False

def save_md5(md5_str):
    """"
        将每一份文件的md5值存储在md5.txt里
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))   # 获取当前文件所在的目录
    md5_path = os.path.join(current_dir, config.md5_path)    # 拼接出md5.txt文件的完整路径
    with open(md5_path,"a",encoding="utf-8") as f:
        f.write(md5_str + "\n")    # 将md5值写入文件，每个md5值占一行
    

def get_string_md5(input_str,encoding="utf-8"):
    """"
        获取字符串的md5值
    """
    # 将字符串编码为字节数组
    str_bytes = input_str.encode(encoding)
    
    # 创建md5对象
    md5_obj = hashlib.md5()
    # 更新md5对象
    md5_obj.update(str_bytes)
    # 获取md5值
    md5_hex = md5_obj.hexdigest()

    return md5_hex

class KnowledgeBaseService:

    def __init__(self):
        # 向量存储的实例Chroma向量库对象
        os.makedirs(config.persist_directory, exist_ok=True)   # 创建存储目录，如果目录已存在则不报错
        self.chroma = Chroma(
            collection_name = config.collection_name,                         # 集合名称，类似数据库的表名
            embedding_function = DashScopeEmbeddings(model = "text-embedding-v4"z),   
            persist_directory= config.persist_directory   # 数据库文件存储路径
        )    

        # 文本分割器的对象
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size = config.chunk_size,                        # 分段的最大长度
            chunk_overlap = config.chunk_overlap,                      # 分段之间的重叠长度，重叠部分可以提供上下文信息，帮助模型更好地理解文本内容
            separators = config.separators,     # 分割文本的依据，优先级递减，直至符合文本长度要求
            length_function = len,                  # 统计长度的指标，默认为字符数
            )
    def upload_by_str(self, content: str, filename):
        """"
            将传入的字符串进行向量化，存入向量数据库中
        """
        # 计算文件的md5值
        md5_hex = get_string_md5(content) 

        if check_md5(md5_hex):
            return f"文件 {filename} 已经存在，上传失败！"
        
        # 文件分割
        if len(content) > config.max_spliter_char_number:
            # 文本分割 -> [str...]
            knowledge_chunks = self.spliter.split_text(content)
        else:
            knowledge_chunks = [content]

        # 向量化并存储到向量数据库中
        metadata = {
            "source" : filename,
            "create_time" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator" : "link"
        }

        self.chroma.add_texts(
            knowledge_chunks, 
            metadatas = [metadata for _ in knowledge_chunks]    # 每一个knowledge_chunk都对应一个metadata，然后封装在一个[]内。
        )

        # 将文件的md5值存储在md5.txt里
        save_md5(md5_hex)

        return f"文件 {filename} 上传成功！"

if __name__ == "__main__":
    service = KnowledgeBaseService()
    service.upload_by_str("周杰伦","test.py")