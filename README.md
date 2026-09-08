#   DocuRAG — PDF Question Answering with RAG

DocuRAG is an AI-powered **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**.

Users can upload a PDF document and ask questions about its content. The application extracts the document text, creates chunks, generates embeddings, stores them in FAISS, retrieves the most relevant information, and uses **Groq's GPT-OSS 120B** model to generate an answer.

##   Live Demo

  **Try the application:**  
[Open DocuRAG](https://docurag-nufaftqopbalgv3eovtn3p.streamlit.app/)

> Replace `PASTE_YOUR_STREAMLIT_APP_LINK_HERE` with your actual Streamlit Cloud URL.

---

##   Features

-   Upload PDF documents
-   Extract text from PDFs
-   Split documents into overlapping chunks
-   Generate text embeddings using Sentence Transformers
-   Perform fast similarity search using FAISS
-   Generate answers using Groq's `openai/gpt-oss-120b`
-   Provide page references when possible
-   Interactive question-answering interface
-   API key stored securely using Streamlit Secrets
-   Reduces hallucination by answering only from retrieved document context

---

##   How It Works

The application follows this RAG pipeline:

```text
             PDF Upload
                 ↓
          Text Extraction
                 ↓
             Chunking
                 ↓
        Sentence Embeddings
                 ↓
              FAISS
          Vector Database
                 ↓
          Similarity Search
                 ↓
       Relevant Document Chunks
                 ↓
          Groq GPT-OSS 120B
                 ↓
          Generated Answer
                 ↓
          Streamlit Interface
