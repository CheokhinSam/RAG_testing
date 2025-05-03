import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
# 環境變量
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("LLM_DEPLOYMENT")

def get_embeddings():
    return AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_API_BASE,
        api_version="2023-05-15"
    )

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=LLM_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_API_BASE,
        api_version="2024-02-15-preview",
        temperature=0.3
    )

llm = get_llm()
embeddings = get_embeddings()

# 設置 FAISS
documents = [
    Document(page_content="The capital of France is Paris.", metadata={"source": "geography"}),
    Document(page_content="Python is a popular programming language.", metadata={"source": "programming"})
]
vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# 保存 FAISS 索引（可選）
vector_store.save_local("faiss_index")

# 定義檢索工具
retriever_tool = create_retriever_tool(
    retriever,
    name="search_documents",
    description="Search for relevant documents based on a query. Useful for answering questions about geography, programming, or other stored knowledge."
)

# 定義自定義工具
@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city."""
    return f"Weather in {city}: Sunny, 25°C"

@tool
def calculate_sum(numbers: list[float]) -> float:
    """Calculate the sum of a list of numbers."""
    return sum(numbers)

# 設置代理
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for RAG tasks. Use the search_documents tool to retrieve relevant information from the database, and use other tools for external queries or calculations. Provide concise and accurate answers."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
tools = [retriever_tool, get_weather, calculate_sum]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# 測試
queries = [
    "What is the capital of Chinese?",
    "What's the weather in Macau?",
    "What is the sum of 10, 20, and 30?",
    "What's the capital of France and the weather there?"
]
for query in queries:
    result = agent_executor.invoke({"input": query})
    print(f"Query: {query}")
    print(f"Answer: {result['output']}\n")