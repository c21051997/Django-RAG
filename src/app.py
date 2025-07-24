# These 3 lines are needed for Streamlit's Cloud environment
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


# Config
# Index from the Pinecone account
INDEX_NAME = "django-docs"
NAMESPACE = "django-docs-namespace"
# OpenAI model we're using to generate responses
LLM_MODEL = "gpt-3.5-turbo"

def format_docs(docs: list[Document]) -> str:
    # A helper function to format the retrieved documents into a single string
    return "\n\n".join(doc.page_content for doc in docs)

# Rag Chain Setup, loads the retriever and language model
# @st.cache_resource tells Streamlit to run this function only once
@st.cache_resource
def load_components():
    print("Loading components...")

    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

    # Collect the APi key from Streamlists secrets manager
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])

    # Initilise the OpenAI LLM
    llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    # Connect to the existing Pinecone vector db
    vector_store = PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings, namespace=NAMESPACE)
    
    # Create a 'retriever', a tools to search the vector db
    retriever = vector_store.as_retriever()

    print("Components loaded successfully.")
    return retriever, llm

# Streamlist UI
# Set the title and icon that appear in the browser tab
st.set_page_config(page_title="Django DocuBot", page_icon="🤖")
# Set the main title on the web page
st.title("Django DocuBot 🤖")

# Load all of the AI components, show error if failure
try:
    retriever, llm = load_components()
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
if user_prompt := st.chat_input("How do I create a model?"):
    # Add the users new message to the jistory and display it
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    st.chat_message("user").markdown(user_prompt)

    # Create the assistant's response
    with st.chat_message("assistant"):
        # Show a "Thinking..." spinner while the RAG chain is working
        with st.spinner("Thinking..."):
            
            # Retrieve documents first
            retrieved_docs = retriever.invoke(user_prompt)
            
            # Format the documents into a single context string
            context = format_docs(retrieved_docs)

            # Create the prompt that will be sent to the LLM
            final_prompt = f"""
            You are an expert assistant for the Django web framework. Answer the user's question based ONLY on the following context.
            If the context doesn't contain the answer, say "I don't have enough information to answer that question."

            CONTEXT:
            {context}

            QUESTION:
            {user_prompt}

            ANSWER:
            """

            # Invoke the LLM with the final prompt
            response = llm.invoke(final_prompt).content
            
            st.markdown(response)

    # Add the assistant's response to the history
    st.session_state.messages.append({"role": "assistant", "content": response})