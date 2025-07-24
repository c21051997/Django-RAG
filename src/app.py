# These 3 lines are needed for Streamlit's Cloud environment
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Config
# Index from the Pinecone account
INDEX_NAME = "django-docs"
# OpenAI model we're using to generate responses
LLM_MODEL = "gpt-3.5-turbo"

#  Rag Chain Setup 
# @st.cache_resource tells Streamlit to run this function only once
@st.cache_resource
def load_components():
    print("Loading components...")
    # Collect the APi key from Streamlists secrets manager
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Connect to the existing Pinecone vector db
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME, 
        embedding=embeddings, 
        pinecone_api_key=st.secrets["PINECONE_API_KEY"])
    
    # Create a 'retriever', a tools to search the vector db
    retriever = vector_store.as_retriever()

    # Initilise the OpenAI LLM
    llm = ChatOpenAI(
        model_name=LLM_MODEL, 
        temperature=0, 
        openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Create the prompt that will be sent to the LLM
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

    # RAG chain that defines the step by step process
    # | "pipes" the output of one step to the next
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Components loaded successfully.")
    return rag_chain

# Streamlist UI
# Set the title and icon that appear in the browser tab
st.set_page_config(page_title="Django DocuBot", page_icon="🤖")
# Set the main title on the web page
st.title("Django DocuBot 🤖")

# Load all of the AI components, show error if failure
try:
    rag_chain = load_components()
except Exception as e:
    st.error(f"Failed to load components: {e}")
    st.stop()

# Define chat history
# st.session_state is Streamlit's way of remembering data accross interactions
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past chat messages on the role
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main loop that waits for the user input in the chat box
if prompt := st.chat_input("How do I create a model?"):
    # Add the users new message to the jistory and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # Create the assistant's response
    with st.chat_message("assistant"):
        # Show a "Thinking..." spinner while the RAG chain is working
        with st.spinner("Thinking..."):
            # Call the RAG chain with the user's prompt to get the answer
            response = rag_chain.invoke(prompt)
            st.markdown(response)
    # Add the assistant's response to the history
    st.session_state.messages.append({"role": "assistant", "content": response})