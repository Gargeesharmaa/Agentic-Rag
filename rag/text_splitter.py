from langchain_text_splitters import RecursiveCharacterTextSplitter
from document_loader import DocumentLoader

class TextSplitter:
    def __init__(self, file_path, chunk_size=1000, chunk_overlap=200):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self):
        loader = DocumentLoader(self.file_path)
        documents = loader.load_document()

        # Fixed: Changed chunk_size=self.chunk_overlap to chunk_size=self.chunk_size
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        final_chunks = text_splitter.split_documents(documents)
        return final_chunks
