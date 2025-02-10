import os

from gradio import ChatInterface
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# 使用smith进行调试
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "liang-lang-smith"
os.environ["LANGSMITH_API_KEY"] = "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"

model = ChatOllama(model="llama3.1")

prompt = SystemMessage(content="""
        你是设计用于与 SQL 数据库交互的代理。
        给定一个输入问题，创建一个语法正确的 Mysql 查询来运行，然后查看查询结果并返回答案。
        除非用户指定他们希望获得的特定示例数量，否则请始终将查询限制为最多 5 个结果。
        你可以按相关列对结果进行排序，以返回数据库中最符合条件的数据。
        切勿查询特定表中的所有列，只询问给定问题的相关列。
        你可以使用与数据库交互的工具。
        只可以使用被允许使用的工具构建最终答案。
        在执行查询之前，你必须仔细检查查询。如果在执行查询时出现错误，请重写查询并重试。

        请勿对数据库执行任何 DML 语句（INSERT、UPDATE、DELETE、DROP 等）。

        首先，你应该查看数据库中的表扫描所有可以查询的内容。
        请勿跳过此步骤。
        然后，你应该查询最相关表的架构。
        最后 你要根据你生成的sql查询出结果数据
        如果查询出错，则根据报错修改你的sql后重试,直到查询出最终的数据
    """)


db_user = "root"
db_password = "Isysc0re123"
db_host = "172.30.1.160"
db_name = "bmos_mes"

sql_database = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

toolkit = SQLDatabaseToolkit(db=sql_database, llm=model)

tools = toolkit.get_tools()

agent = create_react_agent(model, tools, messages_modifier=prompt)

def chat(message, history):
    result = ""
    for chunk in agent.stream({"messages": [HumanMessage(content=message)]}):
        if chunk.get('agent') is not None:
            if chunk['agent'].get('messages') is not None:
                result += chunk['agent'].get('messages')[0].content
                yield result


if __name__ == '__main__':
    gr = ChatInterface(fn=chat, type="messages")
    gr.launch()

