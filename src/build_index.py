import os
import json
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm # (Library for creating a simple progress bar)

# Config 
# Load API keys from environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

INDEX_NAME = "django-docs"
DATA_PATH = "data/django_docs"

def main():
    print("Loading documents...")
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.json'):
            filepath = os.path.join(DATA_PATH, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.append(Document(page_content=data['content'], metadata={'source': data['url']}))

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    print("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    print(f"Uploading {len(chunks)} chunks to Pinecone index '{INDEX_NAME}'...")
    # This will create a new index if it doesn't exist, or use an existing one
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME, pinecone_api_key=PINECONE_API_KEY)

    print("✅ Index built and uploaded successfully!")

if __name__ == "__main__":
    main()