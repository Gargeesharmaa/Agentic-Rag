from typing import List
from langchain_core.documents import Document
from rag.vectorstore import VectorStorageManager

class Retriever:
    """Retrieves the most relevant documents for a user query."""
    def __init__(self):
        self.manager = VectorStorageManager()
        self.vector_searcher = self.manager.load_existing_vectorstore()

    def retrieve(
            self,
            query: str,
            top_k: int = 3,
    ) -> List[Document]:
        """Retrieves relevant documents using semantic vector search."""
        documents = self.vector_searcher.similarity_search(
            query=query,
            k=top_k
        )
        return documents
