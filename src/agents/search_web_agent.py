import dotenv
import os
from typing import List, Dict, Any, Optional
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_perplexity import ChatPerplexity
from azure.identity import DefaultAzureCredential
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver


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
    azure_endpoint=AZURE_OPENAI_ACCOUNT,
    api_version="2025-01-31",
    temperature=0,
    max_tokens=1000
) 

instrucao = """
You are a web search agent. Your task is to search the web for information about {query}"""

# StateGraph and Schema
class SearchWeb(BaseModel):
    query: str # search query
    response: Optional[list[Dict[str, Any]]] # search perplexity response
    answer: Optional[str] # answer to the query

# Generate nodes
def search_web(state: SearchWeb):
    try:
        prompt = state["query"]
        response_schema = List[Dict[str, Any]]
        response = search_model.with_structured_output(response_schema)
        system_message = instrucao.format(query=prompt)
        search_results = response.invoke([SystemMessage(content=system_message)] + [HumanMessage(content="Faça uma busca na web")])
        return {'query': prompt, 'response': search_results}
    except Exception as e:
        print(f"Erro ao buscar informações: {e}")
        return {'query': prompt, 'response': [], 'answer': f"Erro na busca: {str(e)}"}

def analyze_search(state: SearchWeb):
    data = state['response']
    query = state['query']
    prompt = f"""Analise os resultados da busca para a query: {query}. Os resultados são: {data}."""
    # analyze the search results
    response = analyze_model.invoke([SystemMessage(content=prompt)])
    return {'query': query, 'response': data, 'answer': response.content} # return the analysis result

# Define nodes and edges
constructor = StateGraph(SearchWeb)
constructor.add_node("search_web", search_web)
constructor.add_node("analyze_search", analyze_search)
constructor.add_edge(START, "search_web")
constructor.add_edge("search_web", "analyze_search")
constructor.add_edge("analyze_search", END)

# Compile the graph
memory = MemorySaver()
graph = constructor.compile()

# View the graph
graph_png = graph.get_graph(xray=True).draw_mermaid_png()

# Save to file
with open("images/search_web_agent.png", "wb") as f:
    f.write(graph_png)
print("Graph saved as 'images/search_web_agent.png'")


initial_state = {
    "query": "Paraná Clube"  # O campo obrigatório para iniciar a busca
}
resultado = graph.invoke(initial_state)
print(resultado)

