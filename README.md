# ⚡ AI Documentation Assistant for Django

https://django-rag.streamlit.app

An end-to-end Retrieval-Augmented Generation (RAG) application that acts as an AI-powered chatbot for the official Django documentation.  
  
This project was built as a proof-of-concept to demonstrate a modern, scalable, and cloud-native AI architecture that solves a real-world business problem.

---

## 🚧 The Problem: Inefficient Knowledge Discovery

Developer time is a premium asset at any tech company. A large portion of this time is often lost to inefficient knowledge discovery—developers combing through dense documentation to find answers to specific questions.

This challenge becomes even more acute during onboarding, affecting productivity and delivery timelines.

**This project tackles that issue by providing instant, accurate, and context-aware answers pulled directly from trusted documentation.**

---

## ✅ The Solution: An AI Knowledge Assistant

This application delivers a conversational AI interface where developers can ask questions about the Django framework in natural language. It:

- Retrieves the most relevant sections of the official Django documentation.
- Synthesizes a clear and concise response using a Large Language Model (LLM).
- Turns lengthy search efforts into rapid, intelligent interactions.

<!-- Replace with a URL to your app screenshot -->

---

## 🏗️ System Architecture

A lightweight [Streamlit](https://streamlit.io/) application orchestrates all backend operations via cloud services, forming a scalable and API-driven RAG pipeline.

<!-- Replace with a URL to your architecture diagram -->

### 🔄 Workflow Overview

1. **Data Ingestion**  
   A Python script scrapes and processes the Django documentation using BeautifulSoup, splitting it into manageable text chunks.

2. **Indexing**  
   Text chunks are embedded using the OpenAI API and stored in a [Pinecone](https://www.pinecone.io/) vector database.

3. **Retrieval & Generation**  
   - User query is embedded and used to retrieve relevant chunks from Pinecone.  
   - Retrieved context is injected into a prompt template.  
   - Prompt is passed to OpenAI's GPT-3.5, which generates a grounded answer.  
   - Response is streamed back in real-time to the Streamlit interface.

---

## 🧰 Tech Stack

| Category          | Technology                                               |
|-------------------|-----------------------------------------------------------|
| **AI & Orchestration** | LangChain, OpenAI API (Embeddings & LLM), Pinecone       |
| **Application & UI**   | Streamlit, Python                                       |
| **Data Processing**    | BeautifulSoup, `langchain_text_splitters`               |
| **Deployment**         | Streamlit Community Cloud, GitHub                       |

---

## 🛠️ How to Run Locally

Follow the steps below to set up and run the application on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/c21051997/your-repo-name.git
cd your-repo-name
```

### 2. Set Up Environment

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set API Keys

Set your environment variables for OpenAI and Pinecone:

```bash
export OPENAI_API_KEY="sk-..."
export PINECONE_API_KEY="..."
export PINECONE_ENVIRONMENT="..."
```

### 4. Build the Index (One-Time Step)

Run the following to populate your Pinecone index.
Ensure you have created a django-docs index with 1536 dimensions.

```bash
python src/build_index.py
```

### 5. Run the Streamlit App

```bash
streamlit run src/app.py
```

## 📚 Key Learnings

This project demonstrates real-world skills aligned with modern AI engineering:

**End-to-End RAG Architecture**  
Designed and implemented the full pipeline from raw docs to a deployed app.

**Cloud-Native AI Systems**  
Leveraged OpenAI and Pinecone with an API-first mindset to ensure scalability.

**Vector Databases & Embeddings**  
Hands-on experience in generating embeddings and managing vector similarity search.

**Prompt Engineering & Orchestration**  
Used LangChain to orchestrate retrieval, prompt construction, and LLM interaction.