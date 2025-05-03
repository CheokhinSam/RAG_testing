## This is the testing code for my RAG by using RAGAS evaluataion

import os
import logging
from dotenv import load_dotenv
import psycopg2
from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import Annotated, TypedDict
from tenacity import retry, stop_after_attempt, wait_fixed

load_dotenv()
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy)
from datasets import Dataset

# 初始化日志記錄器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加載環境變量
load_dotenv()

# 檢查必要的環境變量
required_env_vars = [
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "EMBEDDING_DEPLOYMENT",
    "LLM_DEPLOYMENT", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD",
    "LANGSMITH_API_KEY"
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

# 環境變量
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("LLM_DEPLOYMENT")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

try:
    client = Client()
except Exception as e:
    logger.error(f"Failed to initialize LangSmith client: {e}")
    raise

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

###############################################################################################################################################################################

def evaluation(datasamples, llm, embeddings, metrics, verbose=True):
    try: custom_dataset = Dataset.from_dict(datasamples)
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        raise

    ragas_llm = LangchainLLMWrapper(llm)

    try:
        result = evaluate(custom_dataset, metrics=metrics, llm=ragas_llm, embeddings=embeddings)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

    df = result.to_pandas()
    if verbose:
        print("\nEvaluation result：")
        print(df)

    return df

if __name__ == "__main__":


    data_samples = {
        'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
        'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
        'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'],
        ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
    }
    
    llm = get_llm()
    embeddings = get_embeddings()
    metrics = [faithfulness, answer_relevancy]
    result_df = evaluation(data_samples, llm, embeddings, metrics, verbose=True)




