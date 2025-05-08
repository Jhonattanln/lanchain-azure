import dotenv
import os
import logging
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_perplexity import ChatPerplexity
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from langgraph.graph import START, StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver

# Configurar logging para melhor observabilidade
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Variáveis
dotenv.load_dotenv()

# Usar DefaultAzureCredential para autenticação segura
try:
    credential = DefaultAzureCredential()
    logger.info("Autenticação com Azure DefaultAzureCredential bem-sucedida")
except Exception as e:
    logger.error(f"Erro de autenticação: {str(e)}")
    raise

os.environ["LANGSMITH_PROJECT"] = "web search" # project name in langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true" # enable tracing

# Carregar variáveis de ambiente com verificação
AZURE_OPENAI_ACCOUNT = os.getenv("AZURE_OPENAI_ACCOUNT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not all([AZURE_OPENAI_ACCOUNT, AZURE_OPENAI_API_KEY, PERPLEXITY_API_KEY]):
    logger.error("Variáveis de ambiente necessárias não estão definidas")
    raise ValueError("Configure todas as variáveis de ambiente necessárias")

# Models
try:
    # Modelo Perplexity para busca na web
    search_model = ChatPerplexity(
        temperature=.1,
        pplx_api_key=PERPLEXITY_API_KEY,
        model="sonar-pro"
    )
    
    # Modelo Azure OpenAI para análise dos resultados
    analyze_model = AzureChatOpenAI(
        azure_deployment="o3-mini",
        azure_endpoint=AZURE_OPENAI_ACCOUNT,
        api_version="2025-01-31",
        temperature=0,
        max_tokens=1000
    )
    logger.info("Modelos inicializados com sucesso")
except Exception as e:
    logger.error(f"Erro ao inicializar modelos: {str(e)}")
    raise

# State usando BaseModel ao invés de TypedDict para melhor validação
class SearchWebAgent(BaseModel):
    query: str
    search_results: List[Dict[str, Any]] = []

# Node com tratamento de erro adequado
sys_message = SystemMessage(content="You are a M&A analyst, that can search the web and summarize the results. Your focus is to find the latest news about brazilian merges and acquisitions. You can create thesis and analyze the results. You will respond in Brazilian Portuguese.")

def search_web_node(state):
    """Search the web for the query in the state and update the state with the results."""
    try:
        # Criando mensagem ao invés de passar string diretamente
        messages = [HumanMessage(content=state.query)]
        logger.info(f"Realizando busca na web para: {state.query}")
        
        # Usar .invoke() ao invés do método depreciado __call__
        response = search_model.invoke(messages)
        
        # Extrair conteúdo da resposta
        if hasattr(response, 'content'):
            search_results = response.content
        else:
            search_results = str(response)
            
        logger.info("Busca na web concluída com sucesso")
        
        # Atualizar o estado com os resultados
        return {"search_results": search_results}
    except AzureError as e:
        logger.error(f"Erro Azure na busca web: {str(e)}")
        return {"search_results": [{"error": str(e)}]}
    except Exception as e:
        logger.error(f"Erro na busca web: {str(e)}")
        return {"search_results": [{"error": str(e)}]}

# Construir o grafo
builder = StateGraph(SearchWebAgent)
builder.add_node('search_web', search_web_node)
builder.add_edge(START, 'search_web')
builder.add_edge('search_web', END)

# Compilar o grafo com tratamento de erro
try:
    graph = builder.compile()
    logger.info("Grafo compilado com sucesso")
except Exception as e:
    logger.error(f"Erro ao compilar grafo: {str(e)}")
    raise

# Criar visualização do grafo
try:
    graph_png = graph.get_graph(xray=True).draw_mermaid_png()
    with open("images/perplexity_agent.png", "wb") as f:
        f.write(graph_png)
    logger.info("Visualização do grafo salva com sucesso")
except Exception as e:
    logger.error(f"Erro ao criar visualização do grafo: {str(e)}")

# Executar o grafo com tratamento de erro adequado
if __name__ == "__main__":
    try:
        logger.info("Iniciando agente de busca web")
        result = graph.invoke(SearchWebAgent(query="Informações sobre o mercado de tilápia no Brasil"))
        
        print("\nResultados da busca:")
        print(result["search_results"])
        
        logger.info("Agente de busca web concluído com sucesso")
    except Exception as e:
        logger.error(f"Erro ao executar agente de busca web: {str(e)}")
        raise
