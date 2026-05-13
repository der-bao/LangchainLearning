import os,json
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self,session_id,storage_path):
        self.session_id = session_id            # 会话id，用于区分不同用户的历史消息
        self.storage_path = storage_path        # 不同会话id的存储文件所在的文件夹
        self.file_path = os.path.join(storage_path, f"{session_id}.json")  # 存储文件的完整路径

        # 确保文件夹存在
        os.makedirs(storage_path, exist_ok=True)

    def add_message(self, message):
        # Sequence[BaseMessage] -> None
        all_messages = list(self.messages)  # 获取当前历史消息列表 self.messages是父类的属性，返回一个生成器对象，转换成列表
        all_messages.append(message)        # 将新消息添加到历史消息列表中

        # 将数据同步写入本地文件
        # 类对象写入文件 -> 一堆二进制
        # 采用message_to_dict将消息对象转换成字典，再将字典写入json文件
        new_messages = [message_to_dict(msg) for msg in all_messages]  # 将消息对象列表转换成字典列表

        # 将字典列表写入json文件
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(new_messages, f,ensure_ascii=False)  # 将字典列表写入json文件

    @property       # @property装饰器将方法转换成属性调用,调用时不需要加括号（简洁），不会直接暴露方法的实现细节（封装）
    def messages(self):
        # 文件内json数据 -> 消息对象列表
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages_dict_list = json.load(f)  # 从json文件中读取字典列表, json -> list[dict]
                return messages_from_dict(messages_dict_list)  # list[dict] -> list[BaseMessage]
        except FileNotFoundError:
            return []  # 如果文件不存在，返回一个空列表，表示没有历史消息
        
    def clear(self):
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([], f,ensure_ascii=False)  # 将一个空列表写入json文件，清空历史消息



def get_history(session_id):
    return FileChatMessageHistory(session_id, storage_path="./data/history")
  