import json
import os
import pandas as pd
from datasets import Dataset

# Import evaluate and metrics
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Import pipeline & free evaluator configs
from agent_engine import agentic_rag_pipeline
from evals.ragas_config import ragas_llm, ragas_embeddings


def run_evaluation():
    # 1. Load Evaluation Dataset
    with open("evals/dataset.json", "r") as f:
        eval_samples = json.load(f)

    results_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    # 2. Execute RAG Pipeline on each sample
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

        # Run pipeline
        output = agentic_rag_pipeline.invoke(inputs)

        results_data["user_input"].append(question)
        results_data["response"].append(output.get("generation", ""))
        results_data["retrieved_contexts"].append(output.get("documents", []))
        results_data["reference"].append(sample.get("reference", ""))

    # 3. Convert to HuggingFace Dataset required by Ragas
    eval_dataset = Dataset.from_dict(results_data)

    # 4. Define metrics list
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    # 5. Run Evaluation (Pass llm and embeddings explicitly here)
    print("🚀 Running Ragas evaluation with Groq...")
    score = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    # 6. Save & Display Results
    df = score.to_pandas()
    os.makedirs("evals/reports", exist_ok=True)
    df.to_csv("evals/reports/experiment_results.csv", index=False)

    print("\n=== Ragas Evaluation Summary ===")
    print(
        df[
            [
                "user_input",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            ]
        ]
    )


if __name__ == "__main__":
    run_evaluation()