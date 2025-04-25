import dotenv
import os
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

dotenv.load_dotenv()
credential = DefaultAzureCredential()

os.environ["LANGSMITH_PROJECT"] = "web search" # project name in langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracing

AZURE_OPENAI_ACCOUNT = os.getenv("AZURE_OPENAI_ACCOUNT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

class ClippingAgent(BaseModel):
    """Clipping agent to search the web and summarize the results."""
    prompt: str 
    response: str
    


# Funcions tools
def web_search(query: str) -> str:
    """Search the web for a query.

    Args:
        query: The query to search for.
    """
    search_web = TavilySearch(
        max_results=5,
        topic='general'
    )
    search_web.invoke({"query": query})

tools = [web_search] # create a list of tools

model = AzureChatOpenAI(model='gpt-4o-mini',
                  api_key=AZURE_OPENAI_API_KEY,
                  api_version='2024-12-01-preview',
                  azure_endpoint=AZURE_OPENAI_ACCOUNT
                  )

llm_with_tools = model.bind_tools(tools)

sys_message = SystemMessage("You are a clipping agent. You will be given a question and you need to search the web and summarize the results.")

# Node

def search_agent(state: ClippingAgent):
    """Search the web for a query and summarize the results."""


builder = StateGraph(ClippingAgent)

builder.add_node("search_agent", search_agent)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'search_agent')
builder.add_conditional_edges('search_agent', tools_condition)
builder.add_edge('tools', 'search_agent')
builder.add_edge('search_agent', END)

react_agent = builder.compile()

graph_png = react_agent.get_graph(xray=True).draw_mermaid_png()

# Save to file
with open("images/search_web.png", "wb") as f:
    f.write(graph_png)

print("Graph saved as 'images/search_web.png'")

config = {"configurable": {"thread_id": "1"}}

react_agent.invoke({"prompt": "Qual é a capital do Brasil?"})

