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

# Load and process the PDF
def load_pdf():
    pdf_path = "Docs/abc.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    
    loader = UnstructuredLoader(pdf_path)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# Generate test dataset using TestsetGenerator
def generate_testset(documents):
    llm = get_llm()
    embeddings = get_embeddings()
    
    # Initialize TestsetGenerator
    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embeddings=embeddings,
        chunk_size=1000  # Adjust based on your needs
    )
    
    # Generate test dataset
    testset = generator.generate(
        documents=documents,
        test_size=10,  # Number of test cases to generate (adjust as needed)
        distributions={"question_type": {"factual": 0.7, "reasoning": 0.3}}  # Customize question types
    )
    
    return testset.to_dataset()


# Main evaluation pipeline
def main():
    # Load PDF and generate test dataset
    try:
        documents = load_pdf()
        test_dataset = generate_testset(documents)
    except Exception as e:
        print(f"Error generating test dataset: {e}")
        print("Falling back to custom dataset")
        

    # Initialize models
    azure_model = get_llm()
    ragas_azure_model = LangchainLLMWrapper(azure_model)
    azure_embeddings = get_embeddings()

    # Metrics
    metrics = [faithfulness, answer_relevancy]

    # Evaluate
    result = evaluate(
        test_dataset,
        metrics=metrics,
        llm=ragas_azure_model,
        embeddings=azure_embeddings
    )

    # Output results
    df = result.to_pandas()
    print(df)

if __name__ == "__main__":
    main()