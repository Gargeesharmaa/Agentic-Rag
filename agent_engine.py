import os
from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
load_dotenv() 

from rag.retrieval import Retriever

# Initialize fundamental elements
document_retriever = Retriever()
llm = ChatGroq(model="llama3-70b-8192", temperature=0)


class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Are documents relevant, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

def retrieve_node(state: GraphState):
    retrieved_docs = document_retriever.retrieve(query=state["question"], top_k=3)
    return {"documents": [d.page_content for d in retrieved_docs], "question": state["question"]}

def grade_documents_node(state: GraphState):
    question = state["question"]
    filtered_docs = []
    grade_prompt = "Question: {q}\nDocument:\n{d}\nIs this relevant? Reply 'yes' or 'no'."
    
    for doc in state["documents"]:
        score = structured_llm_grader.invoke(grade_prompt.format(q=question, d=doc))
        if score.binary_score.lower() == "yes":
            filtered_docs.append(doc)
    return {"documents": filtered_docs, "question": question}

def generate_node(state: GraphState):
    context = "\n\n---\n\n".join(state["documents"])
    prompt = f"Context:\n{context}\n\nQuestion: {state['question']}\nAnswer based strictly on context."
    return {"generation": llm.invoke(prompt).content}

def transform_query_node(state: GraphState):
    prompt = f"Optimize this search query for a vector database: '{state['question']}'. Return only the optimized query."
    return {"question": llm.invoke(prompt).content, "documents": []}

def decide_to_generate(state: GraphState) -> Literal["generate", "transform_query"]:
    return "generate" if state["documents"] else "transform_query"

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("transform_query", transform_query_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"})
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("generate", END)

agentic_rag_pipeline = workflow.compile()


@tool
def agentic_rag_tool(question: str) -> str:
    """
    Retrieves internal document contents and generates answers based on user files, 
    company reports, or knowledge base data. Input must be a specific question string.
    """
    inputs = {"question": question, "documents": [], "generation": ""}
    result = agentic_rag_pipeline.invoke(inputs)
    
    return result.get("generation", "Could not locate relevant documentation answers.")

tools = [agentic_rag_tool]

system_prompt = (
    "You are a versatile corporate assistant powered by Groq.\n"
    "You have access to the 'agentic_rag_tool' to look up internal data.\n"
    "Guidelines:\n"
    "- For normal conversation or logic questions, answer directly.\n"
    "- If asked about company policies, data, or file uploads, route the task to 'agentic_rag_tool'."
)

master_agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=system_prompt
)

query = "Hey there! Can you check our files and tell me what our key risks are for 2026?"
response = master_agent.invoke({"messages": [("human", query)]})

# Print final result
print("\n=== Final Master Agent Output ===")
print(response["messages"][-1].content)
