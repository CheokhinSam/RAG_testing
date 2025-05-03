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
from datasets import Dataset
from ragas.testset import TestsetGenerator  # Import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

load_dotenv()

# Environment variables
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

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.nba.com/stats/players/")

llm = get_llm()
embeddings = get_embeddings()

"""pdf_path = "Docs/cde.pdf"
loader = UnstructuredLoader(pdf_path)"""
documents = loader.load()

generator = TestsetGenerator.from_langchain(
    llm,
    llm,
    embeddings,
)

testset = generator.generate_with_langchain_docs(documents, testset_size=5)

testset.to_dataset()
