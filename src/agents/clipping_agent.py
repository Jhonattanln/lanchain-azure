import dotenv
import os
from langchain_openai import ChatOpenAI
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

## Functions tools
def internet
