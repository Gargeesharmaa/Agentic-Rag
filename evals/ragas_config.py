# evals/ragas_config.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

load_dotenv()

# 1. Free Evaluator LLM via Groq
base_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_KEY")
)

# 2. Free Local Embeddings (MiniLM)
evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Wrap for Ragas
ragas_llm = LangchainLLMWrapper(base_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(evaluator_embeddings)