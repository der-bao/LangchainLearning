import json

py_obj = {
    "name":"林凯",
    "age": 30,
    "gender":"male"
}

json_str = '{"name":"林凯","age": 30,"gender":"male"}'

# 将Python对象转换为JSON字符串
json_str = json.dumps(py_obj, ensure_ascii=False)
print(json_str)  

# 将JSON字符串转换为Python对象
py_obj = json.loads(json_str)
print(py_obj)                # 可以观察到，转换后的py对象的键是单引号