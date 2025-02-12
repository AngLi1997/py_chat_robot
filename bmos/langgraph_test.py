from typing import Annotated

from gradio import ChatInterface
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from typing_extensions import TypedDict

# 对话模型
llm = ChatOllama(model="llama3.1")


class State(TypedDict):
    messages: Annotated[list, add_messages]

# 初始化图
graph_builder = StateGraph(State)

def agent(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 添加智能体
graph_builder.add_node("agent", agent)
graph_builder.add_edge(START, "agent")
graph_builder.add_edge("agent", END)

# 添加对话记忆
memory = MemorySaver()
# 对话session
config = {"configurable": {"thread_id": "1"}}

graph = graph_builder.compile(checkpointer=memory)

graph.get_graph().print_ascii()

def chat(message, history):
    events = graph.stream({"messages": [HumanMessage(message)]}, config, stream_mode="values")
    for event in events:
        if isinstance(event, AIMessage):
            event['messages'][-1].pretty_print()
            yield event['messages'][-1].content


if __name__ == '__main__':
    gr = ChatInterface(fn=chat, type="messages")
    gr.launch()