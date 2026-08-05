from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
import os

class DocumentLoader:
    def __init__(self, file_path):
        # Store the path in the object instance
        self.file_path = file_path

    def load_document(self):
        if self.file_path.startswith(("http://", "https://")):
            loader = WebBaseLoader(self.file_path)
            return loader.load()
        
        my_extension = os.path.splitext(self.file_path)[1].lower()

        if my_extension == ".pdf":
            loader = PyPDFLoader(self.file_path)
        elif my_extension == ".docx":
            loader = Docx2txtLoader(self.file_path)
        elif my_extension == ".txt":
            loader = TextLoader(self.file_path)
        else:
            raise ValueError(f"Unsupported format or invalid path: {self.file_path}")

        return loader.load()