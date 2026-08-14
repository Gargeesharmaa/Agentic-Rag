import json
import os
import sys
from pathlib import Path
import pandas as pd
from datasets import Dataset

# Fix legacy vertexai import path
import langchain_google_vertexai
sys.modules['langchain_community.chat_models.vertexai'] = langchain_google_vertexai

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 1. Import evaluate AND the Metric Classes from ragas.metrics
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

# Import pipeline & ragas configuration
from agent_engine import agentic_rag_pipeline
from evals.ragas_config import ragas_llm, ragas_embeddings

EVALS_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVALS_DIR / "dataset.json"


def run_evaluation():
    # Load Evaluation Dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        eval_samples = json.load(f)

    results_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    # Execute RAG Pipeline on each sample
    for sample in eval_samples:
        question = sample["user_input"]
        file_path = sample.get("file_path")

        inputs = {
            "question": question,
            "file_path": file_path,
            "documents": [],
            "generation": "",
            "retry_count": 0,
        }

        output = agentic_rag_pipeline.invoke(inputs)

        raw_docs = output.get("documents", [])
        retrieved_contexts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in raw_docs
        ]

        results_data["user_input"].append(question)
        results_data["response"].append(output.get("generation", ""))
        results_data["retrieved_contexts"].append(retrieved_contexts)
        results_data["reference"].append(sample.get("reference", ""))

    eval_dataset = Dataset.from_dict(results_data)

    # 2. Instantiate the metric classes directly from ragas.metrics
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    # 3. Run Evaluation
    print("🚀 Running Ragas evaluation with Groq...")
    score = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # Save & Display Results
    df = score.to_pandas()
    report_dir = EVALS_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "experiment_results.csv", index=False)

    print("\n=== Ragas Evaluation Summary ===")
    display_cols = [
        "user_input",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols])


if __name__ == "__main__":
    run_evaluation()