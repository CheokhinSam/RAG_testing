# langsm.py
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

# LangSmith 客戶端初始化
try:
    client = Client()
except Exception as e:
    logger.error(f"Failed to initialize LangSmith client: {e}")
    raise

# 示例數據集
examples = [
    {
        "inputs": {"question": "What is the Concept of Multiple-Hypothesis Belief?"},
        "outputs": {"answer": "Multiple-hypothesis belief is a powerful framework for modeling uncertainty in decision-making and AI. By tracking multiple possible states or explanations, it allows systems to adapt to incomplete or noisy data. However, it requires careful trade-offs between computational efficiency and precision to remain practical in real-world applications."},
    },
    {
        "inputs": {"question": "what is Kalman filter localization?"},
        "outputs": {"answer": "The Kalman filter estimates a robot's position and state by combining predictions from motion models with updates from sensor data. It assumes Gaussian uncertainty and uses a recursive two-step process: prediction (using motion models) and update (using sensor measurements). This efficient sensor fusion method is ideal for real-time localization but may struggle with large uncertainties or multimodal distributions."},
    },
    {
        "inputs": {"question": "What is Potential field path planning?"},
        "outputs": {"answer": "Potential field path planning is a technique in robotics where an artificial field guides the robot toward a goal while avoiding obstacles. The goal acts as an attractive force, pulling the robot closer, while obstacles act as repulsive forces, pushing the robot away. The robot moves by following the gradient of this field, ensuring smooth navigation."},
    }
]

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

WORKING_DIR = 'Docs/'
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

def create_vector_store(file_paths, collection_name="rag_collection", chunk_size=500, chunk_overlap=100):
    embeddings = get_embeddings()
    all_chunked_docs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        loader = UnstructuredLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
        chunked_docs = text_splitter.split_documents(documents)
        all_chunked_docs.extend(chunked_docs)
    
    try:
        db = PGVector.from_documents(
            embedding=embeddings,
            documents=all_chunked_docs,
            collection_name=collection_name,
            connection_string=CONNECTION_STRING,
            pre_delete_collection=False
        )
        logger.info(f"Created vector store for collection '{collection_name}'")
        return db
    except Exception as e:
        logger.error(f"Failed to create vector store: {e}")
        raise

def load_existing_vector_store(collection_name="rag_collection"):
    embeddings = get_embeddings()
    try:
        db = PGVector(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_string=CONNECTION_STRING
        )
        logger.info(f"Loaded existing vector store for collection '{collection_name}'")
        return db
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise

def get_available_collections():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM langchain_pg_collection;")
            collections = [row[0] for row in cur.fetchall()]
        conn.close()
        return collections if collections else ["No collections available"]
    except Exception as e:
        logger.error(f"Error fetching collections: {e}")
        raise

def rerank_documents(query, docs, llm):
    rerank_prompt = """
    You are an expert at evaluating document relevance. 
    Given a query and a document, assign a relevance score between 0 and 10, where 0 means completely irrelevant and 10 means highly relevant. 
    Provide only the numeric score as output, nothing else.

    Query: {query}
    Document: {document}
    Score:
    """
    prompt = ChatPromptTemplate.from_template(rerank_prompt)
    chain = prompt | llm
    
    scored_docs = []
    for doc in docs:
        try:
            response = chain.invoke({"query": query, "document": doc.page_content})
            score = float(response.content.strip())
        except ValueError:
            logger.error(f"Invalid score format from LLM: {response.content}")
            score = 0
        scored_docs.append((doc, score))
    
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs]

def retrieve_docs(db, query, k=8, use_reranking=True):
    initial_k = k + 2 if use_reranking else k
    try:
        initial_docs = db.similarity_search(query, k=initial_k)
        logger.debug(f"Initial docs: {initial_docs}")
    except Exception as e:
        logger.error(f"Failed to retrieve documents: {e}")
        raise

    if use_reranking:
        llm = get_llm()
        reranked_docs = rerank_documents(query, initial_docs, llm)
        final_docs = reranked_docs[:k]
        logger.debug(f"Final docs: {final_docs}")
    else:
        final_docs = initial_docs[:k]
    return final_docs

def reformulate_query(query, llm):
    reformulation_template = """
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

    User Query: {query}
    Reformulated Query:
    """
    reformulation_prompt = ChatPromptTemplate.from_template(reformulation_template)
    reformulation_chain = reformulation_prompt | llm
    try:
        response = reformulation_chain.invoke({"query": query})
        reformulated_query = response.content.strip()
        logger.info(f"Original query: {query} | Reformulated query: {reformulated_query}")
        return reformulated_query
    except Exception as e:
        logger.error(f"Failed to reformulate query: {e}")
        return query

template = """
You are a teaching assistant designed to help students understand concepts clearly and engagingly. Your goal is to provide concise, educational answers that are easy to follow, even for beginners.
Follow these steps to answer the user’s question:
Understand the question: Identify the core concept or problem the user is asking about.
Use relevant information: Combine the retrieved information (if provided) with your knowledge to form a complete and accurate response. If retrieved information is limited or unclear, prioritize your knowledge and note any gaps.
Structure the answer:
Context (if needed): Briefly state the background or context of the question to set the stage, but skip this for simple or straightforward questions.
Key points: Summarize 2–4 main points that address the question, explaining each point in detail. Use simple language and avoid jargon unless it’s essential (e.g., specific terms required by the subject). If jargon is used, define it clearly.
Answer: Provide a clear, direct answer to the question, tying back to the key points.
Enhance with examples: Include 1–2 relevant, simple examples to illustrate key points when they help clarify the concept, but avoid examples for very basic or obvious questions.
Handle uncertainty: If the answer isn’t fully clear from the information available, state “I don’t have enough information to answer completely” and suggest what additional details (e.g., specific context or data) would help.
Keep it concise and engaging: Aim for a response that is thorough but concise (typically 100–300 words, depending on complexity). Use a friendly, conversational tone to maintain student interest.
Adapt the structure flexibly: for simple questions, you may skip the context or combine steps to keep the answer brief. For complex questions, ensure all steps are addressed to provide clarity.

Question: {question}
Context: {context}
Answer:
"""

@retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
@traceable()
def question_file(question, documents=None, memory=None, use_reformulation=True, use_reranking=True):
    model = get_llm()
    if memory is None:
        memory = ConversationBufferMemory()
    
    if use_reformulation:
        reformulated_question = reformulate_query(question, model)
    else:
        reformulated_question = question

    if documents is None:
        db = load_existing_vector_store()
        documents = retrieve_docs(db, reformulated_question, k=8, use_reranking=use_reranking)
    context = "\n".join([doc.page_content for doc in documents])
    logger.info(f"The context is: {context}")

    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = prompt | model | output_parser
    
    if memory:
        previous_context = memory.load_memory_variables({}).get("history", "")
        context = f"{previous_context}\n\n{context}"
    
    try:
        logger.info(f"Invoking Azure GPT-4o with reformulated question: {reformulated_question}")
        response = chain.invoke({"question": reformulated_question, "context": context})
        response_content = response
        if memory:
            memory.save_context({"input": question}, {"output": response_content})
        return {"answer": response_content, "documents": documents}
    except Exception as e:
        logger.error(f"Error with Azure GPT-4o: {e}")
        raise

# 評估器
class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

correctness_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.
Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Return your response in the format:
Explanation: <your reasoning>
Correct: <True or False>
Avoid simply stating the correct answer at the outset."""

grader_llm = get_llm()

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    try:
        # 訪問嵌套的 output 結構
        answer = outputs.get("output", {}).get("answer", "")
        answers = f"""\
QUESTION: {inputs['question']}
GROUND TRUTH ANSWER: {reference_outputs['answer']}
STUDENT ANSWER: {answer}"""
        response = grader_llm.invoke([
            {"role": "system", "content": correctness_instructions},
            {"role": "user", "content": answers}
        ])
        response_text = response.content.strip()
        explanation = response_text.split("Explanation:")[1].split("Correct:")[0].strip()
        correct = response_text.split("Correct:")[1].strip().lower() == "true"
        logger.info(f"Correctness evaluation - Explanation: {explanation}, Correct: {correct}")
        return correct
    except Exception as e:
        logger.error(f"Failed to evaluate correctness: {e}")
        return False

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

relevance_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION
Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Return your response in the format:
Explanation: <your reasoning>
Relevant: <True or False>
Avoid simply stating the correct answer at the outset."""

relevance_llm = get_llm()

def relevance(inputs: dict, outputs: dict) -> bool:
    try:
        answer = outputs.get("output", {}).get("answer", "")
        answer_text = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {answer}"
        response = relevance_llm.invoke([
            {"role": "system", "content": relevance_instructions},
            {"role": "user", "content": answer_text}
        ])
        response_text = response.content.strip()
        explanation = response_text.split("Explanation:")[1].split("Relevant:")[0].strip()
        relevant = response_text.split("Relevant:")[1].strip().lower() == "true"
        logger.info(f"Relevance evaluation - Explanation: {explanation}, Relevant: {relevant}")
        return relevant
    except Exception as e:
        logger.error(f"Failed to evaluate relevance: {e}")
        return False

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

grounded_instructions = """You are a teacher grading a quiz. 
You will be given FACTS and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Return your response in the format:
Explanation: <your reasoning>
Grounded: <True or False>
Avoid simply stating the correct answer at the outset."""

grounded_llm = get_llm()

def groundedness(inputs: dict, outputs: dict) -> bool:
    try:
        answer = outputs.get("output", {}).get("answer", "")
        documents = outputs.get("output", {}).get("documents", [])
        doc_string = "\n\n".join(doc.page_content for doc in documents)
        answer_text = f"FACTS: {doc_string}\nSTUDENT ANSWER: {answer}"
        response = grounded_llm.invoke([
            {"role": "system", "content": grounded_instructions},
            {"role": "user", "content": answer_text}
        ])
        response_text = response.content.strip()
        explanation = response_text.split("Explanation:")[1].split("Grounded:")[0].strip()
        grounded = response_text.split("Grounded:")[1].strip().lower() == "true"
        logger.info(f"Groundedness evaluation - Explanation: {explanation}, Grounded: {grounded}")
        return grounded
    except Exception as e:
        logger.error(f"Failed to evaluate groundedness: {e}")
        return False

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]

retrieval_relevance_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a set of FACTS provided by the student. 
Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met
Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Return your response in the format:
Explanation: <your reasoning>
Relevant: <True or False>
Avoid simply stating the correct answer at the outset."""

retrieval_relevance_llm = get_llm()

def retrieval_relevance(inputs: dict, outputs: dict) -> bool:
    try:
        documents = outputs.get("output", {}).get("documents", [])
        doc_string = "\n\n".join(doc.page_content for doc in documents)
        answer_text = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
        response = retrieval_relevance_llm.invoke([
            {"role": "system", "content": retrieval_relevance_instructions},
            {"role": "user", "content": answer_text}
        ])
        response_text = response.content.strip()
        explanation = response_text.split("Explanation:")[1].split("Relevant:")[0].strip()
        relevant = response_text.split("Relevant:")[1].strip().lower() == "true"
        logger.info(f"Retrieval relevance evaluation - Explanation: {explanation}, Relevant: {relevant}")
        return relevant
    except Exception as e:
        logger.error(f"Failed to evaluate retrieval relevance: {e}")
        return False

def main():
    # 初始化内存
    memory = ConversationBufferMemory()

    # 檢查數據庫集合
    try:
        collections = get_available_collections()
        if "rag_collection" not in collections:
            logger.error("Collection 'rag_collection' not found. Please create it first.")
            return
        db = load_existing_vector_store(collection_name="rag_collection")
        logger.info("Successfully loaded vector store")
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return

    # 創建並上傳數據集到 LangSmith
    dataset_name = "rag_evaluationTT"
    try:
        existing_datasets = client.list_datasets()
        if not any(ds.name == dataset_name for ds in existing_datasets):
            dataset = client.create_dataset(dataset_name=dataset_name)
            for example in examples:
                client.create_example(
                    inputs={"question": example["inputs"]["question"]},
                    outputs={"answer": example["outputs"]["answer"]},
                    dataset_id=dataset.id
                )
            logger.info(f"Created dataset '{dataset_name}' on LangSmith")
        else:
            logger.info(f"Dataset '{dataset_name}' already exists")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return

    # 運行問答
    def run_question(inputs):
        question = inputs["question"]
        try:
            result = question_file(
                question=question,
                documents=None,
                memory=memory,
                use_reformulation=True,
                use_reranking=True
            )
            return {"output": result}  # 包裝為 LangSmith 預期的格式
        except Exception as e:
            logger.error(f"Failed to process question '{question}': {e}")
            return {"output": {"answer": "", "documents": []}}

    # 執行評估
    try:
        results = evaluate(
            run_question,
            data=dataset_name,
            evaluators=[correctness, relevance, groundedness, retrieval_relevance],
            experiment_prefix="rag-doc-evaluation",
            metadata={"version": "LCEL context, gpt-4o"}
        )
        logger.info("Evaluation completed successfully")
        print(results)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()