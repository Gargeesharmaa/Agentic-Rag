import os
from typing import List, Literal, Optional, TypedDict
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent

from rag.retrieval import Retriever

load_dotenv()

DB_URL = os.getenv("DATABASE_URL")

# --- Postgres Checkpointer Setup ---
checkpointer = None

if DB_URL:
    try:
        pool = ConnectionPool(
            conninfo=DB_URL, 
            max_size=10, 
            kwargs={"autocommit": True, "prepare_threshold": 0}
        )
        checkpointer = PostgresSaver(pool)
        with pool.connection() as conn:
            checkpointer.setup()
        print("✅ PostgreSQL checkpointer successfully initialized.")
    except Exception as e:
        print(f"⚠️ Could not connect to PostgreSQL: {e}")
        print("🔄 Falling back to in-memory checkpointer (MemorySaver).")
        checkpointer = MemorySaver()
else:
    print("ℹ️ DATABASE_URL not set in .env. Using MemorySaver.")
    checkpointer = MemorySaver()


# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    retry_count: int
    file_path: Optional[str]  # Added to support custom document paths


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Are documents relevant, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)


def retrieve_node(state: GraphState):
    # Dynamically pass target file_path/collection if provided in state
    file_path = state.get("file_path", None)
    document_retriever = Retriever(file_path=file_path)
    
    retrieved_docs = document_retriever.retrieve(
        query=state["question"], top_k=3
    )
    return {
        "documents": [d.page_content for d in retrieved_docs],
        "question": state["question"],
    }


def grade_documents_node(state: GraphState):
    question = state["question"]
    filtered_docs = []
    grade_prompt = (
        "Question: {q}\nDocument:\n{d}\nIs this relevant? Reply 'yes' or 'no'."
    )

    for doc in state["documents"]:
        score = structured_llm_grader.invoke(
            grade_prompt.format(q=question, d=doc)
        )
        if score.binary_score.lower() == "yes":
            filtered_docs.append(doc)
    return {"documents": filtered_docs, "question": question}


def generate_node(state: GraphState):
    # STRICT FALLBACK if no documents matched or passed grading
    if not state["documents"]:
        return {
            "generation": "Information not provided in the document."
        }

    context = "\n\n---\n\n".join(state["documents"])
    
    # Strict prompt injection directly in generate node
    prompt = f"""You are an expert, precise, and reliable AI assistant specializing in document analysis.

### CORE DIRECTIVES:
1. STRICT GROUNDING: Answer the user's question using ONLY the provided context retrieved from the document.
2. ACCURACY OVER SUPPOSITION: Do not assume, extrapolate, or use outside knowledge.
3. UNANSWERABLE QUESTIONS: If the answer cannot be directly and explicitly found in the retrieved context, respond with EXACTLY: "Information not provided in the document."
4. TONALITY: Maintain a clear, factual, objective, and detailed tone.

### RETRIEVED CONTEXT:
{context}

Question: {state['question']}
"""
    return {"generation": llm.invoke(prompt).content}


def transform_query_node(state: GraphState):
    prompt = f"Optimize this search query for a vector database: '{state['question']}'. Return only the optimized query."
    new_query = llm.invoke(prompt).content.strip()
    return {
        "question": new_query,
        "documents": [],
        "retry_count": state.get("retry_count", 0) + 1,
    }


def decide_to_generate(state: GraphState) -> Literal["generate", "transform_query"]:
    if state.get("retry_count", 0) >= 2:
        return "generate"

    return "generate" if state["documents"] else "transform_query"


# Build the Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("transform_query", transform_query_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {"transform_query": "transform_query", "generate": "generate"},
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

agentic_rag_pipeline = workflow.compile()


@tool
def agentic_rag_tool(question: str, file_path: Optional[str] = None) -> str:
    """Retrieves internal document contents and generates answers based on user files.
    
    Args:
        question: Specific question string to search for.
        file_path: Optional path to a specific document to filter retrieval.
    """
    inputs = {
        "question": question,
        "file_path": file_path,
        "documents": [],
        "generation": "",
        "retry_count": 0,
    }
    result = agentic_rag_pipeline.invoke(inputs)

    return result.get(
        "generation", "Information not provided in the document."
    )


tools = [agentic_rag_tool]

# Master Agent Prompt: Focuses on tool selection and direct pass-through of strict answers
MASTER_SYSTEM_PROMPT = """You are an expert AI assistant for document analysis.

### CORE DIRECTIVES:
1. Always use the `agentic_rag_tool` to search custom documents for user questions.
2. Rely strictly on the tool output to answer the user.
3. If the tool outputs "Information not provided in the document.", output that exact phrase to the user. Do NOT attempt to answer from outside knowledge.
"""

# Create Master Agent using Postgres Checkpointer
master_agent = create_react_agent(
    model=llm, 
    tools=tools, 
    prompt=MASTER_SYSTEM_PROMPT, 
    checkpointer=checkpointer
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test_session_1"}}
    query = "why we need ensemble models?"
    response = master_agent.invoke(
        {"messages": [("human", query)]}, 
        config=config
    )

    print("\n=== Final Master Agent Output ===")
    print(response["messages"][-1].content)