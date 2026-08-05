from text_splitter import TextSplitter
from vectorstore import VectorStorageManager

file_to_process = "your_document.docx" 
splitter = TextSplitter(file_path=file_to_process, chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text()

db_manager = VectorStorageManager(persist_directory="./chroma")
vector_db = db_manager.create_vectorstore(chunks)

print(f"Successfully processed {len(chunks)} chunks into your vector database!")
