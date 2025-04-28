import dotenv
import os
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from azure.identity import DefaultAzureCredential
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

dotenv.load_dotenv()
credential = DefaultAzureCredential()

os.environ["LANGSMITH_PROJECT"] = "web search" # project name in langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracing

AZURE_OPENAI_ACCOUNT = os.getenv("AZURE_OPENAI_ACCOUNT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Define the search tool using the @tool decorator
@tool
def web_search_tool(query: str) -> str:
    """Search the web for information.
    
    Args:
        query: The search query
    
    Returns:
        Search results as text
    """
    search = TavilySearch(max_results=2, topic='news')
    results = search.invoke(query)
    return str(results)

# Set up the LLM
model = ChatOpenAI(model='gpt-4o-mini')

# Bind the tool to the model
llm_with_tools = model.bind_tools([web_search_tool])

# System message for the agent
sys_message = SystemMessage(content='You are a web search agent. You will be given a question and you need to search the web and summarize the results.')

# Define the agent node function with a different name to avoid conflicts
def agent_node(state: MessagesState):
    """Process messages and generate a response with tool usage."""
    messages = [sys_message] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

# Build the agent graph
builder = StateGraph(MessagesState)

# Add nodes to the graph
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode([web_search_tool]))

# Connect the nodes
builder.add_edge(START, "agent")
builder.add_conditional_edges(
    "agent",
    tools_condition
)
builder.add_edge("tools", "agent")
builder.add_edge("agent", END)

# Compile the graph
react_graph = builder.compile()

# Test the agent
if __name__ == "__main__":
    query = "Quartzo Capital"
    message = [HumanMessage(content=query)]
    result = react_graph.invoke({"messages": message})
    print(result)