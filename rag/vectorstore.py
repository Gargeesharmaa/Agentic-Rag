import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStorageManager:
    def __init__(self, persist_directory="./chroma"):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={"device": "cpu"}
        )

    def create_vectorstore(self, document_chunks):
        if not document_chunks:
            raise ValueError("No document chunks provided.")
        
        print(f"Embedding chunks locally using all-MiniLM-L6-v2...")

        vectorstore = Chroma.from_documents(
            documents=document_chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        return vectorstore

    def load_existing_vectorstore(self):
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"No database at {self.persist_directory}")

        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model 
        )
