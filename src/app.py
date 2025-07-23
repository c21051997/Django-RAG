__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Config
# The path to the Chroma vector database we created in build_index.py
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# The name of the local LLM that is being used for generation
# It around the limit of what can be ran on my laptop
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


# Use st.cache_resource to run this function only once, the first time the app loads
# This saves time and resources by not reloading the models on every user interaction.
@st.cache_resource
def load_components():
    # Load the vector store and create the RAG chain
    print("Loading components...")

    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    # Load the Chroma vector store
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # Create the retriever, which is a tool to find relevant documents from the vector store
    # Retrieve top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # # Define the Large Language Model (LLM) from Ollama
    # llm = ChatOllama(model=LLM_MODEL)
    
    # Define the LLM using the Hugging Face Endpoint
    llm = HuggingFaceEndpoint(
        repo_id=HF_LLM_ENDPOINT,
        max_length=128,
        temperature=0.5,
        huggingfacehub_api_token=st.secrets["HF_TOKEN"],
    )
    

    # Define the prompt template
    # This structures how we ask the LLM to behave
    prompt_template = """
    You are an expert assistant for the Django web framework. Answer the user's question based ONLY on the following context.
    If the context doesn't contain the answer, say "I don't have enough information to answer that question."
    Keep your answer concise and include a link to the source document from the metadata.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # This is the RAG chain. It defines the sequence of operations
    # The | symbol (a "pipe") passes the output of one step as input to the next
    rag_chain = (
        # The user's question is passed to the retriever to fetch relevant context
        # The original question is also passed through unchanged
        {"context": retriever, "question": RunnablePassthrough()}
        # The context and question are fed into our prompt template
        | prompt
        # The formatted prompt is sent to the LLM
        | llm
        # The LLM's output is converted to a simple text string
        | StrOutputParser()
    )
    print("Components loaded successfully.")
    return rag_chain

# StreamLit UI
# Set the title and icon for the browser tab
st.set_page_config(page_title="Django DocuBot", page_icon="🤖")
# Set the title of the web page
st.title("Django DocuBot 🤖")
st.write("Ask me anything about the Django documentation!")

# Load the RAG chain
try:
    rag_chain = load_components()
except Exception as e:
    st.error(f"Failed to load components: {e}")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How do I create a model?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get the response from the RAG chain
            response = rag_chain.invoke(prompt)
            st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})