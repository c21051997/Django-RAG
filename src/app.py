__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
INDEX_NAME = "django-docs"
LLM_MODEL = "gpt-3.5-turbo"

# --- RAG CHAIN SETUP ---
@st.cache_resource
def load_components():
    print("Loading components...")
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Connect to the existing Pinecone index
    vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, pinecone_api_key=st.secrets["PINECONE_API_KEY"])
    retriever = vector_store.as_retriever()

    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

    prompt_template = """
    You are an expert assistant for the Django web framework. Answer the user's question based ONLY on the following context.
    If the context doesn't contain the answer, say "I don't have enough information to answer that question."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Components loaded successfully.")
    return rag_chain

# --- STREAMLIT UI ---
st.set_page_config(page_title="Django DocuBot", page_icon="🤖")
st.title("Django DocuBot 🤖")

try:
    rag_chain = load_components()
except Exception as e:
    st.error(f"Failed to load components: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How do I create a model?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})