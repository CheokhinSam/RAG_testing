import os
import logging
import re
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness, context_precision, context_recall
from datasets import Dataset

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

# 自定義文檔加載並添加標題
def load_and_add_headlines(path):
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} document pages from {path}")
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")
        raise
    for doc in docs:
        content = doc.page_content
        headers = re.findall(r'^[^\n]*[A-Z][^\n]*$', content, re.MULTILINE)
        doc.metadata['headlines'] = headers if headers else []
        if 'summary' in doc.metadata:
            del doc.metadata['summary']
        if 'summary_embedding' in doc.metadata:
            del doc.metadata['summary_embedding']
        logger.info(f"Document metadata: {doc.metadata}")
    return docs

# 主邏輯
path = "Docs/cde.pdf"
docs = load_and_add_headlines(path)

llm = get_llm()
embeddings = get_embeddings()

# 生成測試集
generator_llm = LangchainLLMWrapper(llm)
generator_embeddings = LangchainEmbeddingsWrapper(embeddings)
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings,
)

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

# 設置 RAG 系統
logger.info("Setting up RAG system...")
# 分割文檔
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
logger.info(f"Split into {len(chunks)} chunks")

# 創建向量數據庫
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 設置 RAG 提示詞
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say so. Provide a concise and accurate answer.

    **Question**: {question}

    **Context**: {context}

    **Answer**: """
)

# 格式化上下文
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 設置 RAG 鏈（使用 LCEL）
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 運行測試
logger.info("Running RAG evaluation...")
results = []
for _, row in df.iterrows():
    question = row["user_input"]
    ground_truth = row["reference"]
    testset_context = row["reference_contexts"]
    try:
        # 運行 RAG 並獲取上下文
        generated_answer = rag_chain.invoke(question)
        retrieved_docs = retriever.invoke(question)
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "retrieved_context": retrieved_docs,
            "testset_context": testset_context
        })
    except Exception as e:
        logger.error(f"Error processing question '{question}': {str(e)}")
        continue

# 保存測試結果
results_df = pd.DataFrame(results)
results_df.to_csv("rag_evaluation_results.csv", index=False)
logger.info(f"RAG evaluation results saved to {os.path.abspath('rag_evaluation_results.csv')}")

# 評估 RAG
logger.info("Evaluating RAG performance...")
eval_data = {
    "question": results_df["question"],
    "ground_truth": results_df["ground_truth"],
    "answer": results_df["generated_answer"],
    "contexts": results_df["retrieved_context"].apply(lambda x: [doc.page_content for doc in x]),
    "ground_truth_contexts": results_df["testset_context"].apply(lambda x: [x])
}
eval_dataset = Dataset.from_dict(eval_data)

# 使用更新後的指標
metrics = [answer_correctness, faithfulness, context_precision, context_recall]
eval_results = evaluate(
    dataset=eval_dataset,
    metrics=metrics,
    llm=generator_llm,
    embeddings=generator_embeddings
)

# 保存評估結果
print("Evaluation results:", eval_results)
eval_results_df = eval_results.to_pandas()
eval_results_df.to_csv("rag_metrics.csv", index=False)
logger.info(f"RAG metrics saved to {os.path.abspath('rag_metrics.csv')}")