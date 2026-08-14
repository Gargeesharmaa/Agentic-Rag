# ingest.py
from rag.vectorstore import VectorStorageManager

pdf_path = r"C:\Users\welcome\Projects\agentic rag system\ml.pdf"

print("Starting ingestion...")
manager = VectorStorageManager()
manager.create_vectorstore(pdf_path)
print("✅ Ingestion complete! Document chunks stored in ./chroma")