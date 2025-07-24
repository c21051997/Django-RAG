# This must be the first lines of your app.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- CONFIGURATION ---
INDEX_NAME = "django-docs"
NAMESPACE = "django-docs-namespace"
LLM_MODEL = "gpt-3.5-turbo"

# --- HELPER FUNCTION ---
def format_docs(docs: list[Document]) -> str:
    """A helper function to format the retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- RAG COMPONENTS SETUP ---
@st.cache_resource
def load_components():
    """
    Loads the retriever and the language model.
    This function is cached to run only once.
    """
    print("Loading RAG components...")
    
    # Initialize Pinecone and OpenAI components
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    # Load the vector store and create the retriever
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, 
        embedding=embeddings,
        namespace=NAMESPACE
    )
    retriever = vector_store.as_retriever()
    
    print("Components loaded successfully.")
    return retriever, llm

# --- STREAMLIT UI ---
st.set_page_config(page_title="Django DocuBot", page_icon="🤖")
st.title("Django DocuBot 🤖")

try:
    # Load the two main components
    retriever, llm = load_components()
except Exception as e:
    st.error(f"Failed to load AI components: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("How do I create a model?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # This is the new, explicit, and robust process
            
            # 1. Retrieve documents first
            retrieved_docs = retriever.invoke(user_prompt)
            
            # 2. Format the documents into a single context string
            context = format_docs(retrieved_docs)
            
            # 3. Manually create the final prompt
            final_prompt = f"""
            You are an expert assistant for the Django web framework. Answer the user's question based ONLY on the following context.
            If the context is empty or doesn't contain the answer, say "I don't have enough information to answer that question."
            
            CONTEXT:
            {context}
            
            QUESTION:
            {user_prompt}
            
            ANSWER:
            """
            
            # 4. Invoke the LLM with the final prompt
            response = llm.invoke(final_prompt).content
            
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})