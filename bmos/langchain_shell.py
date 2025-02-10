from langchain_community.tools import ShellTool
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

shell = ShellTool()

tools = [Tool(name="Shell", func=shell.run, description="运行shell命令")]

model = ChatOllama(model="llama3.1")

agent = create_react_agent(model, tools)

if __name__ == '__main__':
    result = agent.invoke({"messages": "查看当前所有开放的端口"})
    print(result)