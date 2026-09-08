import os
import hashlib

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="DocuRAG — PDF Q&A",
    page_icon="📚",
    layout="wide",
)

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5


# -----------------------------
# Cached resources
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# -----------------------------
# PDF / RAG helpers
# -----------------------------
def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append({
                "page": page_number,
                "text": text,
            })

    return pages


def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "page": page["page"],
                })

            if end >= len(text):
                break

            start = max(end - overlap, start + 1)

    return chunks


def build_faiss_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


def retrieve_chunks(query, index, chunks, embedding_model, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score),
            })

    return results


def answer_question(question, retrieved_chunks, client):
    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a precise document question-answering assistant.

Answer the user's question using ONLY the supplied document context.
If the answer is not contained in the context, say:
"I couldn't find that information in the uploaded document."

Do not invent facts. Keep the answer clear and useful.
When possible, mention the relevant page number(s).
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1200,
    )

    return completion.choices[0].message.content


# -----------------------------
# UI
# -----------------------------
st.title("📚 DocuRAG")
st.caption(
    "Upload a PDF, build a local FAISS vector index, and ask questions "
    "using an open-weight model served through Groq."
)

with st.sidebar:
    st.header("⚙️ Settings")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.write(f"**Chunk size:** `{CHUNK_SIZE}` characters")
    st.write(f"**Top-K retrieval:** `{TOP_K}`")

    if st.button("🗑️ Clear document"):
        for key in ["file_hash", "pages", "chunks", "index", "messages"]:
            st.session_state.pop(key, None)
        st.rerun()

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="The PDF is processed in memory. Its FAISS index is created for the current app session.",
)

if uploaded_file is None:
    st.info("Upload a PDF to create your RAG knowledge base.")
    st.stop()

# Identify document so the index is rebuilt only when the file changes.
file_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.get("file_hash") != file_hash:
    with st.status("Building your document knowledge base...", expanded=True) as status:
        st.write("1. Extracting PDF text...")
        uploaded_file.seek(0)
        pages = extract_pdf_text(uploaded_file)

        if not pages:
            status.update(label="Could not extract text", state="error")
            st.error(
                "No selectable text was found. This version works with text-based PDFs. "
                "Scanned/image-only PDFs need OCR."
            )
            st.stop()

        st.write("2. Creating overlapping chunks...")
        chunks = chunk_text(pages)

        st.write("3. Creating embeddings...")
        embedding_model = load_embedding_model()

        st.write("4. Building FAISS vector index...")
        index = build_faiss_index(chunks, embedding_model)

        st.session_state.file_hash = file_hash
        st.session_state.pages = pages
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.messages = []

        status.update(
            label=f"Knowledge base ready — {len(chunks)} chunks",
            state="complete",
        )

chunks = st.session_state["chunks"]
index = st.session_state["index"]
embedding_model = load_embedding_model()

col1, col2 = st.columns(2)
with col1:
    st.metric("PDF pages", len(st.session_state["pages"]))
with col2:
    st.metric("Text chunks", len(chunks))

st.divider()

client = get_groq_client()

if client is None:
    st.warning("Groq API key is missing.")
    st.code('GROQ_API_KEY = "your_groq_api_key_here"', language="toml")
    st.info(
        "For Streamlit Cloud, add GROQ_API_KEY in your app's Secrets. "
        "Never commit the API key to GitHub."
    )
    st.stop()

st.subheader("💬 Ask questions about your PDF")

for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about the uploaded document...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating an answer..."):
            retrieved = retrieve_chunks(
                question,
                index,
                chunks,
                embedding_model,
                TOP_K,
            )

            answer = answer_question(
                question,
                retrieved,
                client,
            )

        st.markdown(answer)

        with st.expander("🔎 Retrieved context"):
            for i, item in enumerate(retrieved, start=1):
                st.markdown(
                    f"**Result {i} — Page {item['page']} — "
                    f"Similarity: {item['score']:.3f}**"
                )
                st.write(item["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })
