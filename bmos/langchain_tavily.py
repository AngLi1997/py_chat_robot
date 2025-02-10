import os

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM

# 使用smith进行调试
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "liang-lang-smith"
os.environ["LANGSMITH_API_KEY"] = "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"

os.environ["TAVILY_API_KEY"] = "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"

# 基本模型
model = OllamaLLM(model="deepseek-r1:latest")

# 联网搜索
search = TavilySearchResults(max_results=2)

# 绑定工具
tools = [search]



if __name__ == '__main__':
    result = model.invoke([
        HumanMessage(content="成都的天气怎么样")
    ])
    print(result)