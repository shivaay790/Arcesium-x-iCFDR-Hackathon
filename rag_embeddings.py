import os
import sys
from langchain_community.vectorstores import FAISS  
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


PSYCH_CONTENT_DIR = "rag_database/"
FAISS_DB_DIR = "rag_database/faiss_index"  

def initialize_vector_db():
    """Initialize or create the local vector database of psychological content using FAISS"""
    
    # Create directories if they don't exist
    if not os.path.exists(PSYCH_CONTENT_DIR):
        os.makedirs(PSYCH_CONTENT_DIR)
        print("Created content directory.")
    
    # Load PDF documents (create a directory and add as many PDF files for your toolkit of agent leveraging RAG method.
    print("Loading PDF documents...")
    loader = DirectoryLoader(PSYCH_CONTENT_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("ERROR: No documents found in the directory.")
        print(f"Please add PDF files to {PSYCH_CONTENT_DIR} directory before running this script.")
        return None
    
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize FAISS vector store
    print(f"Initializing FAISS vector database at '{FAISS_DB_DIR}'...")
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )
    
    # Save the FAISS index to disk
    print(f"Saving FAISS index to {FAISS_DB_DIR}...")
    vector_store.save_local(FAISS_DB_DIR)
    
    print(f"Vector database created locally with {len(chunks)} text chunks")
    return vector_store

if __name__ == "__main__":
    print("=== RAG Database Creation Tool (FAISS Version) ===")
    print("This tool will process PDF files and create local vector embeddings in FAISS.")
    print(f"Source directory: {PSYCH_CONTENT_DIR}")
    print(f"Target FAISS database directory: {FAISS_DB_DIR}")
    
    vector_db = initialize_vector_db()
    
    if vector_db:
        print("\nDatabase creation complete!")
        print(f"Your embeddings are stored locally in: {FAISS_DB_DIR}")
    else:
        print("\nDatabase creation failed.")
        print("Please ensure you have PDF files in the rag_database folder.")
