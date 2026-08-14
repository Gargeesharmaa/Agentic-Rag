# 🤖 Agentic RAG — Intelligent Document Analysis & Agentic RAG

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)]() [![Python](https://img.shields.io/badge/python-3.9%2B-yellow)]()

Effective nutshell

Agentic RAG is a compact, production-ready Retrieval-Augmented Generation (RAG) platform designed for trustworthy, auditable question-answering over documents. It combines agentic orchestration, multi-LLM generation, and a robust retrieval/evaluation pipeline so teams can build systems that answer questions with grounded evidence, full session history, and reproducible checkpoints.

Project summary

Agentic RAG enables developers and analysts to:
- Ingest diverse documents (PDF, DOCX, TXT, CSV) into a vector store.
- Retrieve and rank relevant passages using configurable retrievers (Chroma/FAISS/Pinecone).
- Use agentic logic to transform queries and retry when retrieval is insufficient.
- Orchestrate one or more LLMs (Groq, OpenAI, Vertex AI, local HF models) to produce grounded answers.
- Persist sessions, checkpoints, and trace data in PostgreSQL for replay, audit, and evaluation (RAGAS).

Why use Agentic RAG

- Grounded answers: responses cite and rely on retrieved document content, reducing hallucinations.
- Agentic resilience: automated query transformation and retry improves answer coverage.
- Reproducibility: Postgres-backed session & checkpoint store makes results auditable.
- Flexible deployment: swap vector stores and models to match cost, latency, and accuracy needs.

Quick start

1) Clone & create a virtual environment

```bash
git clone https://github.com/Gargeesharmaa/Agentic-Rag.git
cd Agentic-Rag
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env-example .env
```

2) Configure `.env` (example keys)

```
GROQ_API_KEY=your_groq_key
DATABASE_URL=postgresql://agentic_user:secure_password@localhost:5432/agentic_rag
LANGSMITH_API_KEY=your_langsmith_key  # optional
OPENAI_API_KEY=your_openai_key  # optional
VECTOR_STORE=chroma  # or faiss, pinecone
```

3) Start PostgreSQL (Docker recommended)

```bash
docker run --name agentic-rag-db \
  -e POSTGRES_USER=agentic_user \
  -e POSTGRES_PASSWORD=secure_password \
  -e POSTGRES_DB=agentic_rag \
  -p 5432:5432 -d postgres:15-alpine
```

4) Run the app

```bash
python app.py
# Open http://127.0.0.1:8000 in your browser
```

Files of interest

- rag/retrieval.py — vector store selection & retrieval pipeline
- agent_engine.py — agent orchestration and LLM selection
- ingestion/ — document ingestion and chunking utilities
- evals/ — RAGAS evaluation harness and visualization scripts

Demo & assets

This README embeds previews and links to demo assets located in `assets/`. Use the raw URLs below if you need direct access or to embed the media elsewhere:

- Demo photo (thumbnail): https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo_photo.png
- Demo video (playable): https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo_video.mp4
- Code generation preview: https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/Code_Generated_Image.png
- Evaluation graph (example): https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/eval_graph.png

Preview (click to open demo video):

<p align="center">
  <a href="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo_video.mp4">
    <img src="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo_photo.png" alt="Demo photo" width="720" />
  </a>
</p>

Playable video (in supported renderers):

```html
<video controls width="720" poster="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo_photo.png">
  <source src="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo_video.mp4" type="video/mp4" />
  Your browser does not support the video tag. Click to download: <a href="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo_video.mp4">demo</a>
</video>
```

Code-generation preview image (used in documentation):

<p align="center">
  <img src="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/Code_Generated_Image.png" alt="Code generation preview" width="720" />
</p>

Evaluation with RAGAS

Agentic RAG integrates RAGAS to evaluate faithfulness, precision, recall, and relevancy. The evaluation harness supports batch scoring and visualizations. Example aggregated metrics (illustrative):

| Metric | Score | Interpretation |
|--------|------:|----------------|
| Faithfulness | 0.89 | Responses grounded in source content |
| Context Precision | 0.80 | Retrieved passages are relevant |
| Answer Relevancy | 0.66 | Answers are generally useful |
| Context Recall | 0.67 | Retrieval finds ~2/3 of relevant content |

Run evaluations and visualize results:

```bash
python evals/run_ragas_eval.py --config evals/config.yaml --out results/eval.json
python evals/plot_results.py results/eval.json --out assets/eval_graph.png
```

Interpretation & best practices

- Use a held-out evaluation set to avoid overfitting retrieval heuristics.
- Compare multiple LLMs with the same retrieval outputs to isolate model variance.
- Tune chunk size and overlap to balance context relevance vs. noise.

Pipeline & workflow

High-level pipeline when a user submits a query:

1. User submits query (web UI or API)
2. Retriever fetches top-k document chunks from the vector store
3. Relevance scorer ranks/filters retrieved chunks
4. If relevance is sufficient → LLM(s) generate grounded answer with citations
5. If relevance is insufficient → Agent applies transform (rephrase/expand) and retries
6. Final answer and supporting context are saved to Postgres as a session and checkpoint
7. Optionally, evaluation traces are recorded and scored by RAGAS

Mermaid flowchart (renders on platforms that support Mermaid):

```mermaid
flowchart TD
  A[User query] --> B[Retriever (Chroma/FAISS/Pinecone)]
  B --> C[Relevance Scorer]
  C -->|sufficient| D[LLM(s) generate answer]
  C -->|insufficient| E[Agent transform & retry]
  E --> B
  D --> F[Save session & checkpoints (Postgres)]
  F --> G[Evaluation (RAGAS) & Visualization]
```

Detailed workflow notes

- Retriever: configurable vector DB with embeddings; tune top_k and distance metric.
- Relevance scorer: combines embedding similarity with heuristics (keyword overlap, date filters).
- Agent transforms: may expand the query, add clarifying constraints, or switch retrieval filters.
- LLM orchestration: parallel or sequential calls to different models with majority/score-based selection.
- Storage: sessions and checkpoints include raw messages, retrieved contexts, model outputs, and evaluation metadata.

API examples

Single-query invocation

```python
from agent_engine import master_agent
config = {"configurable": {"thread_id": "user_session_123"}}
response = master_agent.invoke({"messages":[("human","What is in the document?")]}, config=config)
print(response["messages"][-1].content)
```

Multi-turn example

```python
messages = [("human","First question?"), ("assistant","Response..."), ("human","Follow-up?")]
response = master_agent.invoke({"messages": messages}, config=config)
```

Configuration tips

- Switch vector stores by editing `rag/retrieval.py`.
- Change default LLM in `agent_engine.py`.
- Use `LANGSMITH_TRACING=true` to capture verbose traces for LangGraph/LangChain integrations.

Troubleshooting

- ModuleNotFoundError → pip install -r requirements.txt
- DB connection refused → ensure Docker/Postgres is running and `DATABASE_URL` is correct
- MemoryError → reduce document size, lower chunk sizes, or increase system resources

Testing & contributing

1. Fork -> create a branch -> implement changes -> open a PR
2. Run tests & formatters locally:

```bash
pip install -r requirements-dev.txt
pytest
black .
flake8 .
```

Support & contact

- Issues: https://github.com/Gargeesharmaa/Agentic-Rag/issues
- Email: gargee6548@gmail.com

License

MIT — see the LICENSE file.

Made with ❤️ by Gargee Sharma
