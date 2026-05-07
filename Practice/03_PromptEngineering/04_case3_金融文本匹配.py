from openai import OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("LLM_API_KEY")
base_url = os.getenv("LLM_BASE_URL")
model_id = os.getenv("LLM_MODEL_ID")

# 定义客户端
client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# 示例数据
examples_data = {
    "是": [
        ("公司ABC发布了季度财报，显示盈利增长。", "财报披露，公司ABC利润上升。"),
        ("公司ITCAST发布了年度财报，显示盈利大幅度增长。", "财报披露，公司ITCAST更赚钱了。")
    ],
    "不是": [
        ("黄金价格下跌，投资者抛售。", "外汇市场交易额创下新高。"),
        ("央行降息，刺激经济增长。", "新能源技术的创新。")
    ]
}

questions = [
    ("利率上升，影响房地产市场。", "高利率对房地产有一定的冲击。"),
    ("油价大幅度下跌，能源公司面临挑战。", "未来智能城市的建设趋势越加明显。"),
    ("股票市场今日大涨，投资者乐观。", "持续上涨的市场让投资者感到满意。")
]

# 角色设定
messages = [
    {"role": "system", "content": f"你帮我完成文本匹配，我给你2个句子，被[]包围，你判断它们是否匹配，回答是或不是，请参考如下示例："},
]

# 添加示例数据到消息列表
for label, sentences in examples_data.items():
    messages.append({"role": "user", "content": f"句子1：[{sentences[0]}], 句子2：[{sentences[1]}]"})
    messages.append({"role": "assistant", "content": label})

# 添加提问数据到消息列表
for question in questions:
    response = client.chat.completions.create(
        model = model_id,
        messages = messages + [{"role": "user", "content": f"句子1：[{question[0]}], 句子2：[{question[1]}]"}],
    )
    print(response.choices[0].message.content)
