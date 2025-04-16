import dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from azure.identity import DefaultAzureCredential
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode

dotenv.load_dotenv()

os.environ["LANGSMITH_PROJECT"] = "internet_agent" # project name in langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracing

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Function

def web_search(query: str) -> str:
    """Search the web for a query.

    Args:
        query: The query to search for.
    """
    tool = TavilySearch(
        max_results=5,
        topic='general'
    )
    tool.invoke({"query": query})

tool = [web_search] # create a list of tools
llm = ChatOpenAI(model='gpt-4o-mini')

llm_with_tools = llm.bind_tools(tool)

# Default system message
sys_msg = SystemMessage(content="""
                        You are a web search assistant. You can use the tools to search in the web an create a clipping of Quartzo Capital.
                        You will response in brazilian portuguese.
                        """)

# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Create a state graph
builder = StateGraph(MessagesState)

# Define the nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tool", ToolNode(tool))

builder.add_edge(START, "assistant")

builder.add_edge("tool", "assistant")
react_graph = builder.compile()

graph_png = react_graph.get_graph(xray=True).draw_mermaid_png()

# Save to file
with open("images/clipping_agent.png", "wb") as f:
    f.write(graph_png)

print("Graph saved as 'images/clipping_agent.png'")

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

messages = [HumanMessage(content="Quais as novas noticias da Quartzo Capital")]
messages = react_graph_memory.invoke({"messages": messages},config)

for m in messages['messages']:
    m.pretty_print()
