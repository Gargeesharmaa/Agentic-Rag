import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.vectorstore import VectorStorageManager
from rag.document_loader import DocumentLoader 

def process_and_ingest_document(file_path: str, persist_directory: str = "./chroma"):
    """
    Dynamically loads, chunks, and ingests a custom document into Chroma DB.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    print(f"--- Loading document: {file_path} ---")
    loader = DocumentLoader(file_path)
    docs = loader.load_document()

    print("--- Chunking document ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    print(f"--- Ingesting {len(chunks)} chunks into vector store ---")
    db_manager = VectorStorageManager(persist_directory=persist_directory)
    vector_db = db_manager.create_vectorstore(chunks)

    print(f"Successfully processed {len(chunks)} chunks from '{os.path.basename(file_path)}'!")
    return vector_db