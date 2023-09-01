from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

DATA_DIR = "data/"
DB_FAISS_DIR = "vectorstores/faiss_db"

# create vector database

def create_vector_db():
    loader = DirectoryLoader(DATA_DIR, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})    
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_DIR)
    
if __name__ == '__main__':
    create_vector_db()
