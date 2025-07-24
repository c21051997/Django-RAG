# This must be the first lines of your app.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# --- CONFIGURATION ---
INDEX_NAME = "django-docs"
NAMESPACE = "django-docs-namespace"

st.set_page_config(page_title="Debug Retriever", page_icon="🔍")
st.title("Pinecone Retriever Debugger 🔍")
st.write("This app tests the connection to Pinecone and shows the raw retrieved documents.")

try:
    # --- INITIALIZE COMPONENTS ---
    st.write("Initializing Pinecone client...")
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    
    st.write("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    st.write("Connecting to Pinecone index...")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, 
        embedding=embeddings,
        namespace=NAMESPACE
    )
    retriever = vector_store.as_retriever()
    st.success("Components initialized successfully!")

except Exception as e:
    st.error(f"An error occurred during initialization: {e}")
    st.stop()


# --- DEBUGGING UI ---
prompt = st.text_input("Enter a question to test the retriever:")

if prompt:
    st.write(f"Searching for documents related to: '{prompt}'")
    with st.spinner("Retrieving documents..."):
        try:
            # Invoke the retriever to get relevant documents
            retrieved_docs = retriever.invoke(prompt)
            
            st.subheader("Retrieved Documents:")
            if not retrieved_docs:
                st.warning("The retriever found 0 documents.")
            else:
                st.info(f"The retriever found {len(retrieved_docs)} documents.")
                for i, doc in enumerate(retrieved_docs):
                    with st.expander(f"Document {i+1} - Source: {doc.metadata.get('source', 'N/A')}"):
                        st.write(doc.page_content)

        except Exception as e:
            st.error(f"An error occurred during retrieval: {e}")