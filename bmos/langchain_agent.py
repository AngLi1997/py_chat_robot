import os

from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM, ChatOllama
from langgraph.prebuilt import chat_agent_executor

# 使用smith进行调试
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "liang-lang-smith"
os.environ["LANGSMITH_API_KEY"] = "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"

os.environ["TAVILY_API_KEY"] = "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"


# 模型
model = ChatOllama(model="llama3.1")

# 联网搜索工具
tools = [TavilySearchResults(max_results=2)]

# 代理
agent_executor = chat_agent_executor.create_tool_calling_executor(model, tools)


if __name__ == '__main__':
    result = agent_executor.invoke({"messages": [HumanMessage(content="今天是哪年几月几号")]})
    print(result)