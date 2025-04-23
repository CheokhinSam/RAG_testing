#models.py
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import logging
import re
from datetime import datetime
import psycopg2  
from langsm import Client, traceable
from typing_extensions import Annotated, TypedDict
from langchain.prompts import PromptTemplate
from langsm import traceable
from langsm import Client

load_dotenv()

langsmith_client = Client()

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



def rewrite_query(original_query):

    rewrite_llm = get_llm()

    reformulation_prompt_template = """
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

    User Query: {original_query}
    Reformulated Query:
    """

    query_rewrite_prompt = ChatPromptTemplate.from_template(reformulation_prompt_template)

    query_rewriter = query_rewrite_prompt | rewrite_llm

    response = query_rewriter.invoke(original_query)
    reformulated_query = response.content
    return reformulated_query

original_query = "What is gyroscope?"
rewritten_query = rewrite_query(original_query)
print("Original query:", original_query)
print("\nRewritten query:", rewritten_query)

@traceable(run_type="chain")
def reformulate_query(query):
    llm = get_llm()
    reformulation_prompt = """
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

    User Query: {query}
    Reformulated Query:
    """
    prompt = ChatPromptTemplate.from_template(reformulation_prompt)
    chain = prompt | llm
    response = chain.invoke({"query": query})
    abcabc = response.content.strip()
    return abcabc

riginal_query = "gyroscope?"
rewritten_query = reformulate_query(riginal_query)
print("\nOriginal query:", riginal_query)
print("\nRewritten query:", rewritten_query)