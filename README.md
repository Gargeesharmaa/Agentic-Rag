# 🤖 Agentic RAG System
### Intelligent Document Analysis with LangGraph & PostgreSQL Backend

**Quick Start:** `pip install -r requirements.txt && cp .env-example .env && python app.py`

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API & Usage](#api--usage)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Agentic RAG System** is a production-ready Retrieval-Augmented Generation (RAG) platform powered by:
- **LangGraph** for agentic orchestration
- **PostgreSQL** for persistent conversation history
- **LangChain** ecosystem for LLM integration
- **Gradio** for intuitive web interface
- **RAGAS** for evaluation metrics

This system enables intelligent document analysis through multi-turn conversations with full state management and conversation history tracking.

### 📊 Performance Overview
Based on the evaluation metrics using RAGAS framework:
- **Faithfulness:** 0.89 (89%) - High accuracy in grounded responses
- **Context Precision:** 0.80 (80%) - Excellent document relevance scoring
- **Answer Relevancy:** 0.66 (66%) - Reliable answer generation
- **Context Recall:** 0.67 (67%) - Effective document retrieval

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multi-LLM Support** | Groq, OpenAI, Google Generative AI, Vertex AI |
| 💾 **Persistent Storage** | PostgreSQL-backed conversation logs & state management |
| 📄 **Document Processing** | PDF, TXT, DOCX, CSV file upload & ingestion |
| 🔄 **Agentic Workflow** | LangGraph-based intelligent routing & retry logic |
| 📊 **Evaluation Ready** | Built-in RAGAS evaluation framework |
| 🌐 **Web Interface** | Gradio-based responsive UI |
| 🔌 **LangSmith Integration** | End-to-end tracing & debugging |
| ⚡ **Session Management** | Multi-session support with history recovery |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Gradio Web Interface                       │
│        (Session Management | Document Upload | Chat)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     app.py (Orchestrator)                    │
│    (Session Control | History Management | DB Operations)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  agent_engine.py (Agent)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LangGraph Workflow:                                  │  │
│  │  Retrieve → Grade Documents → Generate/Transform     │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  PostgreSQL  │  │ ChromaDB/    │  │  LLM APIs    │
│  (History &  │  │ Vector Store │  │  (Groq/OAI)  │
│   Checkpts)  │  │  (Retrieval) │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Workflow Pipeline
```
User Query
    │
    ▼
[Retrieve] → Fetch relevant documents from vector store
    │
    ▼
[Grade Documents] → Evaluate document relevance to query
    │
    ├─ YES (Relevant) → [Generate] → Return answer
    │
    └─ NO (Irrelevant) → [Transform Query] → Retry
         (Max 2 retries)
```

---

## 📋 Prerequisites

### System Requirements
- **Python:** 3.9+ (3.11+ recommended)
- **PostgreSQL:** 12.0+
- **RAM:** Minimum 4GB (8GB+ recommended)
- **Storage:** Minimum 2GB free disk space

### Required API Keys
- **Groq API Key** (Free tier available) - for LLM inference
- **LangSmith API Key** (Optional) - for tracing & debugging
- **Hugging Face Token** (Optional) - for model downloads

---

## 🚀 Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/Gargeesharmaa/Agentic-Rag.git
cd Agentic-Rag
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4: Setup PostgreSQL Database

#### Option A: Local Installation
```bash
# macOS (using Homebrew)
brew install postgresql@15
brew services start postgresql@15

# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql

# Windows
# Download PostgreSQL installer from https://www.postgresql.org/download/windows/
```

#### Option B: Docker Installation (Recommended)
```bash
docker pull postgres:15-alpine
docker run --name agentic-rag-db \
  -e POSTGRES_USER=agentic_user \
  -e POSTGRES_PASSWORD=secure_password \
  -e POSTGRES_DB=agentic_rag \
  -p 5432:5432 \
  -d postgres:15-alpine
```

#### Verify Database Connection
```bash
psql -U agentic_user -d agentic_rag -h 127.0.0.1 -p 5432
```

### Step 5: Environment Configuration
```bash
cp .env-example .env
```

Edit `.env` with your credentials (see [Configuration](#configuration) section).

---

## ⚙️ Configuration

### Environment Variables (.env)

```ini
# ================== LLM Configuration ==================
GROQ_API_KEY=your_groq_api_key_here
GROQ_KEY=your_groq_api_key_here              # For evaluation

# ================== LangSmith Tracing ==================
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=Agentic-RAG-Assistant

# ================== Database Configuration ==================
DATABASE_URL=postgresql://agentic_user:secure_password@localhost:5432/agentic_rag
DB_USER=agentic_user
DB_PASSWORD=secure_password
DB_NAME=agentic_rag
DB_HOST=127.0.0.1
DB_PORT=5432

# ================== Optional: Additional LLM Providers ==================
OPENAI_API_KEY=your_openai_key_here          # For OpenAI models
GOOGLE_API_KEY=your_google_key_here          # For Google Generative AI
GOOGLE_CLOUD_PROJECT=your_gcp_project_id     # For Vertex AI
HF_TOKEN=hf_your_huggingface_token          # For model downloads
```

### Configuration Options Breakdown

| Variable | Purpose | Example |
|----------|---------|---------|
| `GROQ_API_KEY` | Primary LLM inference | `gsk_xxxxxxxxxxxx` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `LANGSMITH_TRACING` | Enable/disable tracing | `true` or `false` |
| `DB_HOST` | Database hostname | `localhost` or `127.0.0.1` |

---

## ▶️ Running the Application

### Standalone Execution
```bash
# Start the Gradio application
python app.py

# Application will be available at http://127.0.0.1:8000
```

### Docker Execution (Full Stack)
```bash
# Build and run with docker-compose
docker-compose up --build

# Access the application at http://localhost:8000
```

### Production Deployment with Gunicorn
```bash
pip install gunicorn
gunicorn --workers 4 --worker-class sync --bind 0.0.0.0:8000 app:demo
```

---

## 💻 API & Usage

### Web Interface (Gradio)

#### Main Chat Interface
1. **New Chat Session** - Click "➕ New Chat Session" button
2. **Type Query** - Enter your question in the input field
3. **Send Message** - Press Enter or click "Send" button
4. **View Response** - Agent processes and displays answer

#### Document Upload
1. Click **📄 Add Context Documents**
2. Select file (.pdf, .txt, .docx, .csv)
3. Click "Upload Document"
4. Document is indexed for retrieval

#### Session Management
1. View past conversations in **📂 Past Conversations** dropdown
2. Click session to load history
3. Continue conversation or start new session

### Python API Usage

```python
from agent_engine import master_agent

# Single query
config = {"configurable": {"thread_id": "user_session_123"}}
response = master_agent.invoke(
    {"messages": [("human", "What is in the document?")]},
    config=config
)
print(response["messages"][-1].content)

# Multi-turn conversation
messages = [
    ("human", "First question?"),
    ("assistant", "Response..."),
    ("human", "Follow-up question?"),
]
response = master_agent.invoke(
    {"messages": messages},
    config=config
)
```

---

## 📊 Performance Metrics

### Evaluation Results (RAGAS Framework)

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **Faithfulness** | 0.89 | 89% of responses grounded in context |
| **Context Precision** | 0.80 | 80% of retrieved docs are relevant |
| **Answer Relevancy** | 0.66 | 66% answer relevance to query |
| **Context Recall** | 0.67 | 67% of relevant docs successfully retrieved |

### Response Quality
- **Latency:** < 5s (typical)
- **Throughput:** ~10-15 concurrent sessions
- **Memory Usage:** ~2GB per active session

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. PostgreSQL Connection Error
```
Error: could not connect to server: Connection refused
```
**Solution:**
```bash
# Check PostgreSQL service status
sudo systemctl status postgresql

# Start PostgreSQL
sudo systemctl start postgresql

# Or use Docker
docker-compose up -d db
```

#### 2. API Key Authentication Failed
```
Error: Invalid API key provided to Groq
```
**Solution:**
```bash
# Verify .env file
cat .env | grep GROQ_API_KEY

# Obtain key from https://console.groq.com/keys
# Update .env with correct key
```

#### 3. Module Import Error
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution:**
```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Or update pip
pip install --upgrade pip
```

#### 4. Memory Error During Document Processing
```
MemoryError: Unable to allocate X GiB for array
```
**Solution:**
```bash
# Process smaller documents
# Or increase system RAM
# Reduce batch size in retrieval config
```

#### 5. Database Lock/Timeout
```
Error: database is locked
```
**Solution:**
```bash
# Restart PostgreSQL
docker-compose restart db

# Or check active connections
psql -U agentic_user -d agentic_rag -c "SELECT * FROM pg_stat_activity;"
```

### Debug Mode

```bash
# Enable verbose logging
LANGSMITH_TRACING=true python app.py

# Check logs
tail -f logs/app.log
```

---

## 📈 Advanced Configuration

### Custom Vector Store Setup
Edit `rag/retrieval.py` to use different vector stores:

```python
# Chroma (Default)
from langchain_chroma import Chroma

# Pinecone
from langchain_pinecone import Pinecone

# FAISS
from langchain_community.vectorstores import FAISS
```

### LLM Model Selection
```python
# In agent_engine.py, change model:
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
    temperature=0
)
```

### Batch Processing
```bash
python ingest.py  # Process multiple documents at once
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup
```bash
pip install -r requirements.txt pytest pytest-cov black flake8
pytest  # Run tests
black .  # Format code
flake8 .  # Lint code
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support & Contact

- **Issues:** [GitHub Issues](https://github.com/Gargeesharmaa/Agentic-Rag/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Gargeesharmaa/Agentic-Rag/discussions)
- **Email:** gargee6548@gmail.com

---

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) - Framework for LLM applications
- [LangGraph](https://langgraph.js.org/) - Agentic orchestration
- [Groq](https://groq.com/) - Fast LLM inference
- [PostgreSQL](https://www.postgresql.org/) - Persistent storage
- [Gradio](https://gradio.app/) - Web interface framework

---

**Built by gargee sharma ❤️ **
