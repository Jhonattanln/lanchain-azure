import dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_tavily import TavilySearch
from azure.identity import DefaultAzureCredential
#from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel, Field

dotenv.load_dotenv()

credential = DefaultAzureCredential()

os.environ["LANGSMITH_PROJECT"] = "internet_agent" # project name in langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracing

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AZURE_INFERENCE_ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ACCOUNT = os.getenv("AZURE_OPENAI_ACCOUNT")

### Tools function

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

tool = [web_search] # create a list of tools


### Model from Azure AI Foundry
model = AzureChatOpenAI(model='gpt-4o-mini',
                  api_key=AZURE_OPENAI_API_KEY,
                  api_version='2024-12-01-preview',
                  azure_endpoint=AZURE_OPENAI_ACCOUNT
                  )

llm_with_tools = model.bind_tools(tool)


### State graph
class ClippingAgent(BaseModel):
    """Clipping agent to search the web and summarize the results."""
    model
    prompt: str = Field(
        description="The prompt to use for the agent.",
        default="You are a clipping agent. You will be given a question and you need to search the web and summarize the results."
    )
    tools: list = Field(
        description="The tools to use for the agent.",
        default=tool
    )
    # Tavily results list
    tavily_search: list[TavilySearch] = Field(
        default_factory=list
    )
    summary: str = Field(
        description="The summary of the search results.",
        default="You are a clipping agent. You will be given a question and you need to search the web and summarize the results."
    )

### Agents

# Search instructions
PROMPT_ENGINEER = f"""
You are a prompt engineer. You will be given a question and you need to format it to be used in a search engine. For this topic {{question}}
"""

instrucions = SystemMessage(content=PROMPT_ENGINEER)

def search_instructions(state: ClippingAgent):
    """Format the query to be used in a search engine."""
    # Get the question from the state
    question = state.prompt
    # Format the question to be used in a search engine
    formatted_question = PROMPT_ENGINEER.format(question=question)
    # Return the formatted question
    return formatted_question
    # Add the formatted question to the state
    state.prompt = formatted_question

# Search the web
def search_web(state: ClippingAgent):
    """Search the web for the query."""
    # Get the question from the state
    search_tool = 
    question = state.prompt
    # Search the web for the question
    search_results = {"tavily_search": [llm_with_tools.invoke([instrucions] + state.tavily_search)]}
    # Add the search results to the state
    state.tavily_search.append(search_results)
    # Return the search results
    return search_results

# Summarize the search results
def summarize_search_results(state: ClippingAgent):
    """Summarize the search results."""
    # Get the search results from the state
    search_results = state.tavily_search
    # Summarize the search results
    summary = model.invoke({"query": search_results})
    # Add the summary to the state
    state.summaty = summary
    # Return the summary
    return summary

# Create the state graph
state_graph = StateGraph(ClippingAgent)

# Add the nodes to the state graph
state_graph.add_node("search_instructions", search_instructions)
state_graph.add_node("search_web", search_web)
state_graph.add_node("summarize_search_results", summarize_search_results)
state_graph.add_edge(START, "search_instructions")
state_graph.add_edge("search_instructions", "search_web")
state_graph.add_edge("search_web", "summarize_search_results")
state_graph.add_edge("summarize_search_results", END)

# Compiler
react_graph = state_graph.compile()

# Visualize the state graph
graph_png = react_graph.get_graph(xray=True).draw_mermaid_png()

# Save to file
with open("images/clipping_agent.png", "wb") as f:
    f.write(graph_png)

print("Graph saved as 'images/clipping_agent.png'")



