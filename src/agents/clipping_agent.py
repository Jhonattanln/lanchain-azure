import dotenv
import os
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_tavily import TavilySearch
from azure.identity import DefaultAzureCredential
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from pydantic import BaseModel, Field

dotenv.load_dotenv()

credential = DefaultAzureCredential()

os.environ["LANGSMITH_PROJECT"] = "internet_agent" # project name in langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracing

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AZURE_INFERENCE_ENDPOINT = os.getenv("AZURE_INFERENCE_ENDPOINT")

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
model = AzureAIChatCompletionsModel(
    endpoint=AZURE_INFERENCE_ENDPOINT,
    credential=credential,
    model_name='phi-4'
)

llm_with_tools = model.bind_tools(tool)


### State graph
class ClippingAgent(BaseModel):
    """Clipping agent to search the web and summarize the results."""
    prompt: str = Field(
        description="The prompt to use for the agent.",
        default="You are a clipping agent. You will be given a question and you need to search the web and summarize the results."
    )
    tools: list = Field(
        description="The tools to use for the agent.",
        default=tool
    )
    memory: MemorySaver = Field(
        description="The memory to use for the agent.",
        default=MemorySaver()
    )
    # Tavily results list
    tavily_search: list[TavilySearch] = Field(
        description="The search engine to use for the agent.",
        default=[]
    )
    summaty: str = Field(
        description="The summary of the search results.",
        default="You are a clipping agent. You will be given a question and you need to search the web and summarize the results."
    )




### Agents

# Search instructions
PROMPT_ENGINEER = """
You are a prompt engineer. You will be given a question and you need format it to be used in a search engine.  
"""

instrucions = SystemMessage(content=PROMPT_ENGINEER)

def search_instructions(question: str) -> str:
    """Format the query to be used in a search engine.

    Args:
        query: The query to format.
    """
    return model.invoke(
        [
            instrucions,
            HumanMessage(content=question)
        ]
    ).content



