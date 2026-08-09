import os
from typing import List, Optional
from langchain_core.documents import Document
from rag.vectorstore import VectorStorageManager

class Retriever:
    """Retrieves the most relevant documents for a user query."""
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Accepts an optional file_path to filter vector searches by a specific document.
        """
        self.file_path = file_path
        self.manager = VectorStorageManager()
        self.vector_searcher = self.manager.load_existing_vectorstore()

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Document]:
        """Retrieves relevant documents using semantic vector search with optional source filtering."""
        search_kwargs = {"k": top_k}
        
        if self.file_path:
            normalized_path = os.path.normpath(self.file_path)
            search_kwargs["filter"] = {"source": normalized_path}

        documents = self.vector_searcher.similarity_search(
            query=query,
            **search_kwargs
        )
        return documents