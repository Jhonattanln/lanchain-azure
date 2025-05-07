import dotenv
import os
from typing import List, Dict, Any
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchian_perplexity import ChatPerplexity
from azure.identity import DefaultAzureCredential
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage


dotenv.load_dotenv()
credential = DefaultAzureCredential()

os.environ["LANGSMITH_PROJECT"] = "web search" # project name in langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracing

AZURE_OPENAI_ACCOUNT = os.getenv("AZURE_OPENAI_ACCOUNT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Models
search_model = ChatPerplexity( # perplexity model to search the web
    temperature=.1,
    pplx_api_key=PERPLEXITY_API_KEY,
    model="sonar-pro"
)

analyze_model = AzureChatOpenAI( # openai model to analyze the search results
    azure_deployment="o3-mini",
    api_version="2025-01-31",
    temperature=0,
    max_tokens=1000
) 


# StateGraph and Schema
class SearchWeb(BaseModel):
    query: str = # search query
    response: list[Dict[str, Any]] # search perplexity response
    answer: str # answer to the query

# Generate nodes
def search_web(state: BaseModel):
    prompt = state["query"]
