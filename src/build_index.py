import os
import json

# Langchain: framework for building applications with LLMs
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm # (Library for creating a simple progress bar)

# Config 
# Load the secret API keys from the local environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Index from the Pinecone account
INDEX_NAME = "django-docs"
# Folder where the scraped data is stored
DATA_PATH = "data/django_docs"

def main():
    # This function builds the entire vector index
    print("Loading documents...")
    documents = []
    # Load all the scraped documentation from the JSON files
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.json'):
            filepath = os.path.join(DATA_PATH, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                # Each file is loaded into a LangChain document object
                # Stores the text content and source URL as metadata
                data = json.load(f)
                documents.append(Document(page_content=data['content'], metadata={'source': data['url']}))

    # Spliting the documents into smaller chunks to make searching easier
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Set up connection to OpenAI's API to create embeddings
    print("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Convert all text chunks into vecotr embeddings and upload to Pinecone
    print(f"Uploading {len(chunks)} chunks to Pinecone index '{INDEX_NAME}'...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME, pinecone_api_key=PINECONE_API_KEY)

    print("✅ Index built and uploaded successfully!")

if __name__ == "__main__":
    main()