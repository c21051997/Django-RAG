import os
import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm # (Library for creating a simple progress bar)

# Config 
# Path to the folder containing the scraped JSON files
DATA_PATH = "data/django_docs"
# Path to where the vector db will be stored
DB_PATH = "chroma_db"
# The hugging face modelfor creating text embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Settings for splitting text into chunks
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
# Process 32 chunks at a time
BATCH_SIZE = 32 

def load_documents():
    # Loads documents from the specified data path
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.json'):
            filepath = os.path.join(DATA_PATH, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Create a LangChain Document object, which stores the text (page_content) and its source (metadata)
                doc = Document(page_content=data['content'], metadata={'source': data['url']})
                documents.append(doc)
    print(f"Loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    # Splits documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def build_vector_store(chunks):
    # Builds the vector store from the document chunks with a progress bar
    print("Initializing embedding model in CPU mode... This may take a moment.")
    
    # Initialize the embedding model, and force it to run on the CPU for stability
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs
    )

    print(f"Creating vector store in batches of {BATCH_SIZE}...")
    
    # Initialize a ChromaDB vector store that will save data to the specified directory
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    # Process the chunks in batches to manage memory and show a progress bar
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding documents"):
        batch = chunks[i:i + BATCH_SIZE]
        # Convert the batch of text chunks into vectors and store them
        vector_store.add_documents(documents=batch)

    print("Vector store created and saved successfully!")
    return vector_store

def main():
    # Main function to build the RAG index
    docs = load_documents()
    chunks = split_documents(docs)
    build_vector_store(chunks)

if __name__ == "__main__":
    main()