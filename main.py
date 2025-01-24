import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# 提示词
prompt_template = """
重要！你是一个严格的数据接口，必须遵守下面几个规则：
- 今年是2025年
- 接口的业务和生物制药、献血、血浆有关
- 不使用自然语言
- 只能使用JSON格式返回，仅包含以下字段：
   - api
   - condition
- 只能返回一个接口
- 目前可以使用的接口有以下这么多，返回的接口必须包含在其中
    {api_list}
- 返回文本格式直接使用{{"api": [接口信息], "condition": [条件(可忽略)]}}
- 当没有匹配的接口时，直接返回null
- 根据日期查询时，condition中不要出现当前年份，根据年份差值计算即可

近义词：
    用户 浆员 献血浆者 志愿者

示例：1
输入请求：查询血浆列表
响应：{{"api": "/user/list"}}
示例：2
输入请求：查询张三的信息
响应：{{"api": "/user/getByName", "condition":"name='张三'"}}
示例：3
输入请求：你是谁
响应："我现在还在学习"
根据上面的规则处理以下的输入请求
输入请求：{input}
"""
prompt = PromptTemplate(input_variables=["input", "api_list"], template=prompt_template)
llm = OllamaLLM(model="llama3.1", temperature=0.1)
chain = prompt | llm
api_list ="""
    /user/list
    /user/getByName
    /plasma/list
    /product/list
    /product/id
    """

def chat(message, history):
    result = ""
    for chunk in chain.stream({"input": message, "api_list": api_list}):
        result += chunk
        yield result


if __name__ == '__main__':
    demo = gr.ChatInterface(fn=chat, type="messages")
    demo.launch()
