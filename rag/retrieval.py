import os
from typing import List, Optional
from langchain_core.documents import Document
from rag.vectorstore import VectorStorageManager


class Retriever:
    """Retrieves the most relevant documents for a user query."""

    def __init__(self, file_path: Optional[str] = None):
        """Accepts an optional file_path to filter vector searches by a specific document."""
        self.file_path = file_path
        self.manager = VectorStorageManager()
        
        # Load the vectorstore instance safely
        try:
            self.vector_searcher = self.manager.load_existing_vectorstore()
        except Exception:
            # If no store exists and a file is passed, create or load via manager method
            if self.file_path and os.path.exists(self.file_path):
                self.vector_searcher = self.manager.create_vectorstore(self.file_path)
            else:
                raise

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
    ) -> List[Document]:
        """Retrieves relevant documents using semantic vector search with flexible metadata matching."""
        
        # 1. First attempt: Search using file_path or filename metadata filtering
        if self.file_path:
            filename = os.path.basename(self.file_path)
            normalized_path = os.path.normpath(self.file_path)
            
            # Try full normalized path filter
            try:
                documents = self.vector_searcher.similarity_search(
                    query=query,
                    k=top_k,
                    filter={"source": normalized_path}
                )
                if documents:
                    return documents
            except Exception:
                pass

            # Try base filename filter
            try:
                documents = self.vector_searcher.similarity_search(
                    query=query,
                    k=top_k,
                    filter={"source": filename}
                )
                if documents:
                    return documents
            except Exception:
                pass

        # 2. Fallback: Direct similarity search across all vectorstore documents
        return self.vector_searcher.similarity_search(
            query=query,
            k=top_k
        )