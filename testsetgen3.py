import os
import logging
import re
from langchain_community.document_loaders import DirectoryLoader
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加載環境變量
load_dotenv()

# 環境變量
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
LLM_DEPLOYMENT = os.getenv("LLM_DEPLOYMENT")

# 獲取嵌入模型
def get_embeddings():
    return AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_OPENAI_API_BASE,
        api_version="2023-05-15"
    )

# 獲取語言模型
def get_llm():
    return AzureChatOpenAI(
        azure_deployment=LLM_DEPLOYMENT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_API_BASE,
        api_version="2024-02-15-preview",
        temperature=0.3
    )

"""# 自定義文檔加載並添加標題
def load_and_add_headlines(path):
    loader = DirectoryLoader(path, glob="**/*.md")
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents")
    for doc in docs:
        content = doc.page_content
        headers = re.findall(r'^(#+ .+)$', content, re.MULTILINE)
        doc.metadata['headlines'] = headers if headers else []
        if 'summary' in doc.metadata:
            del doc.metadata['summary']
        if 'summary_embedding' in doc.metadata:
            del doc.metadata['summary_embedding']
        logger.info(f"Document metadata: {doc.metadata}")
    return docs"""
from langchain_community.document_loaders import PyPDFLoader
def load_and_add_headlines(path):
    # 使用 PyPDFLoader 來加載單個 PDF 文件
    loader = PyPDFLoader(path)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} document pages from {path}")
    
    for doc in docs:
        content = doc.page_content
        # 提取可能的標題（假設標題是每頁中以大寫或特定格式開頭的行）
        headers = re.findall(r'^[^\n]*[A-Z][^\n]*$', content, re.MULTILINE)
        doc.metadata['headlines'] = headers if headers else []
        # 移除不需要的元數據
        if 'summary' in doc.metadata:
            del doc.metadata['summary']
        if 'summary_embedding' in doc.metadata:
            del doc.metadata['summary_embedding']
        logger.info(f"Document metadata: {doc.metadata}")
    return docs

# 主邏輯
"""path = "Sample_Docs_Markdown/"
docs = load_and_add_headlines(path)"""

path = "Docs/cde.pdf"
docs = load_and_add_headlines(path)

llm = get_llm()
embeddings = get_embeddings()

generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

# 初始化 TestsetGenerator
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
)

# 生成並保存測試集
try:
    dataset = generator.generate_with_langchain_docs(docs, testset_size=5)
    logger.info("Dataset generated successfully.")
    df = dataset.to_pandas()
    print("DataFrame columns:", df.columns)
    print("DataFrame preview:\n", df.head())
    output_path = "testset_output.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {os.path.abspath(output_path)}")
except Exception as e:
    logger.error(f"Error during testset generation: {str(e)}")
    raise