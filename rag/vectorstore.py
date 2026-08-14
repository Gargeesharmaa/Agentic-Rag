import os
from typing import List, Union
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStorageManager:
    def __init__(self, persist_directory="./chroma"):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

    def create_vectorstore(self, input_data: Union[str, List[Document], List[str]]):
        """Creates or adds documents to Chroma vector storage.
        
        Accepts:
        - A string file path (e.g., 'ml.pdf')
        - A list of Document objects
        - A list of raw strings
        """
        document_chunks: List[Document] = []

        # 1. Handle file path input (e.g., PDF or TXT)
        if isinstance(input_data, str) and os.path.exists(input_data):
            file_path = input_data
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding="utf-8")
            
            raw_docs = loader.load()
            
            # Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            document_chunks = text_splitter.split_documents(raw_docs)
            
            # Attach source metadata tag for retrieval filtering
            filename = os.path.basename(file_path)
            norm_path = os.path.normpath(file_path)
            for chunk in document_chunks:
                chunk.metadata["source"] = norm_path
                chunk.metadata["filename"] = filename

        # 2. Handle list inputs (Document objects or raw strings)
        elif isinstance(input_data, list):
            for item in input_data:
                if isinstance(item, str):
                    document_chunks.append(Document(page_content=item))
                elif hasattr(item, "page_content"):
                    document_chunks.append(item)

        if not document_chunks:
            raise ValueError(f"No valid document chunks or readable file provided from input: {input_data}")

        print(f"Embedding {len(document_chunks)} chunks locally using all-MiniLM-L6-v2...")

        # 3. Create & persist vectorstore
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