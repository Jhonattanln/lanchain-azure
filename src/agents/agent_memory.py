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

os.environ["LANGSMITH_PROJECT"] = "memory agent test" # project name in langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracing

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

credential = DefaultAzureCredential()


## Functions tools

def multiply(a: int, b: int) -> int:
    """Multiplique a e b.

    Args:
        a: primeiro int
        b: segundo int
    """
    return a * b

# Vamos criar uma ferramenta
def add(a: int, b: int) -> int:
    """Soma a e b.

    Args:
        a: primeiro int
        b: segundo int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a por b.

    Args:
        a: primeiro int
        b: segundo int
    """
    return a / b

tools = [add, multiply, divide] # create a list of tools
llm = ChatOpenAI(model='gpt-4o-mini')

llm_with_tools = llm.bind_tools(tools)

# Default system message
sys_msg = SystemMessage(content="You are a mathematical assistant. You can use the tools to do math operations. You will response in brazilian portuguese.")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


# Graph
builder = StateGraph(MessagesState)

# Define the nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define the arrests: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # Se a última mensagem (resultado) da assistente é uma chamada de ferramenta -> tools_condition roteia para ferramentas
    # Se a última mensagem (resultado) da assistente não é uma chamada de ferramenta -> tools_condition roteia para END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Mostra o grafo
graph_png = react_graph.get_graph(xray=True).draw_mermaid_png()

# Save to file
with open("src/images/agent_graph.png", "wb") as f:
    f.write(graph_png)

print("Graph saved as 'src/images/agent_graph.png'")

memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

messages = [HumanMessage(content="Soma 3 e 4.")]
messages = react_graph_memory.invoke({"messages": messages},config)
#for m in messages['messages']:
#    m.pretty_print()

messages = [HumanMessage(content="Multiplique isso por 2.")]
messages = react_graph_memory.invoke({"messages": messages}, config)
for m in messages['messages']:
    m.pretty_print()
