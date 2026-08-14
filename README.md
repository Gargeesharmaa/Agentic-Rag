# 🤖 Agentic RAG — Intelligent Document Analysis (Nutshell)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)]() [![Python](https://img.shields.io/badge/python-3.9%2B-yellow)]()

A lightweight, production-ready Retrieval-Augmented Generation (RAG) platform with agentic orchestration. Agentic RAG combines LangGraph agents, a PostgreSQL-backed conversation store, and multi-LL[...]

Quick start:
```
pip install -r requirements.txt
cp .env-example .env
python app.py
```

---

## 🚀 Project Nutshell

Agentic RAG helps you ask questions about documents and get grounded, multi-turn answers while preserving full conversation history. It:
- Retrieves relevant document context from a vector store,
- Grades and filters those documents,
- Generates answers via one or more LLMs,
- Supports retries and agentic decision-making (transform/ retry),
- Stores session history and checkpoints in PostgreSQL for reproducibility.

---

## ✨ Highlights & Use Cases

- Multi-LLM orchestration (Groq, OpenAI, Vertex AI, local HF models)
- Persistent session history & checkpointing (Postgres)
- Document ingestion: PDF, DOCX, TXT, CSV
- Gradio web UI for interactive demos
- Built-in RAGAS evaluation and visualization

---

## 🧱 Tech Stack

- Python 3.9+
- LangGraph (agent orchestration)
- LangChain ecosystem (LLM + retrieval integrations)
- PostgreSQL (persistent history & checkpoints)
- Vector store (Chroma / FAISS / Pinecone interchangeable)
- Gradio (web UI)
- RAGAS (evaluation)

---

## ⚙️ Setup & Local Quick Start

1. Clone & prepare
```bash
git clone https://github.com/Gargeesharmaa/Agentic-Rag.git
cd Agentic-Rag
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
cp .env-example .env
```

2. Configure `.env` (example keys)
```
GROQ_API_KEY=your_groq_key
DATABASE_URL=postgresql://agentic_user:secure_password@localhost:5432/agentic_rag
LANGSMITH_API_KEY=your_langsmith_key  # optional
OPENAI_API_KEY=your_openai_key  # optional
```

3. Start PostgreSQL
- Docker (recommended)
```bash
docker run --name agentic-rag-db \
  -e POSTGRES_USER=agentic_user \
  -e POSTGRES_PASSWORD=secure_password \
  -e POSTGRES_DB=agentic_rag \
  -p 5432:5432 -d postgres:15-alpine
```

4. Run the app
```bash
python app.py
# Open http://127.0.0.1:8000
```

---

## 🧩 Configuration Tips

- To switch vector stores, edit rag/retrieval.py (Chroma/FAISS/Pinecone).
- To change the default LLM, update agent_engine.py (example: ChatGroq model selection).
- Use LANGSMITH_TRACING=true for verbose traces.

---

## 📂 Assets (Demo, Screenshot & Eval Graph)

I've embedded thumbnails that link to the demo video and included raw GitHub asset URLs so they render correctly on GitHub.

Preview image (click to open demo video):

<p align="center">
  <a href="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo.mp4">
    <img src="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/screenshot.png" alt="Demo screenshot" width="720" />
  </a>
</p>

Embed a playable video (GitHub may not natively autoplay large files — you can link or use a GIF preview):

```html
<!-- Raw video URL (replace branch/main and filenames as needed) -->
<a href="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo.mp4">Watch demo video</a>

<!-- Optional HTML video tag (works in some renderers): -->
<video controls width="720" poster="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/screenshot.png">
  <source src="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo.mp4" type="video/mp4" />
  Your browser does not support the video tag. Click to download: <a href="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/demo.mp4">demo</a>
</video>
```

Evaluation graph preview:

<p align="center">
  <img src="https://raw.githubusercontent.com/Gargeesharmaa/Agentic-Rag/main/assets/eval_graph.png" alt="Evaluation graph" width="720" />
</p>

Note: If you prefer direct raw URLs to embed from GitHub, use:
https://raw.githubusercontent.com/<owner>/<repo>/main/assets/demo.mp4
(Replace <owner>/<repo>/main and filenames accordingly.)

---

## 📊 Evaluation (RAGAS) — Summary

We evaluate model outputs with the RAGAS framework. Example aggregated metrics:

| Metric | Score | Interpretation |
|--------|-------:|----------------|
| Faithfulness | 0.89 | Responses grounded in source content |
| Context Precision | 0.80 | High-quality retrieval relevance |
| Answer Relevancy | 0.66 | Useful answers, room to improve |
| Context Recall | 0.67 | Retrieval finds 2/3 relevant docs |

Re-run evaluations:
```bash
python evals/run_ragas_eval.py --config evals/config.yaml --out results/eval.json
# visualize
python evals/plot_results.py results/eval.json --out assets/eval_graph.png
```

Include the produced `assets/eval_graph.png` in this README (see Assets section).

---

## 🧪 How the Pipeline Works (Brief)

1. User query → Retrieval from vector store
2. Documents graded for relevance
3. If relevant → Generate grounded answer
4. If not → Transform query & retry (max 2 retries)
5. Save session & checkpoints to Postgres for audit & replay

---

## 🧰 API Snippets

Simple single-query invocation:
```python
from agent_engine import master_agent
config = {"configurable": {"thread_id": "user_session_123"}}
response = master_agent.invoke({"messages":[("human","What is in the document?")]}, config=config)
print(response["messages"][-1].content)
```

Multi-turn example:
```python
messages = [("human","First question?"), ("assistant","Response..."), ("human","Follow-up?")]
response = master_agent.invoke({"messages": messages}, config=config)
```

---

## 🧭 Troubleshooting & Tips

- "ModuleNotFoundError" → pip install -r requirements.txt
- DB connection refused → ensure Docker/Postgres is running and env DATABASE_URL is correct
- MemoryError: process smaller documents or increase system resources

---

## 🤝 Contributing

1. Fork -> branch -> commit -> push -> PR
2. Run tests & formatters:
```bash
pip install -r requirements-dev.txt
pytest
black .
flake8 .
```

---

## 📞 Support & Contact

- Issues: https://github.com/Gargeesharmaa/Agentic-Rag/issues
- Email: gargee6548@gmail.com

---

## 📝 License

MIT License — see the LICENSE file.

---

Made with ❤️ by gargee sharma
