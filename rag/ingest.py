
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.vectorstore import VectorStorageManager
from rag.document_loader import DocumentLoader 

loader = DocumentLoader("./ml_book.pdf")
docs = loader.load_document()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

db_manager = VectorStorageManager(persist_directory="./chroma")
vector_db = db_manager.create_vectorstore(chunks)

print(f"Successfully processed {len(chunks)} chunks into your vector database!")