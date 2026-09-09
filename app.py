import hashlib
import os
from typing import Dict, List, Tuple

import faiss
import pymupdf
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------
APP_TITLE = "HR Policy Assistant"
MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
TOP_K = 5
MAX_CONTEXT_CHARS = 18000


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📘",
    layout="wide",
)

st.title("📘 HR Policy Assistant")
st.caption(
    "Upload an HR Policy PDF and ask questions. "
    "Answers are grounded in the uploaded document using RAG."
)


# -----------------------------
# Cached model loading
# -----------------------------
@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_pages(pdf_bytes: bytes) -> List[Dict]:
    """Extract text page-by-page so retrieved chunks retain page numbers."""
    pages = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

    return pages


# -----------------------------
# Chunking
# -----------------------------
def split_into_chunks(
    text: str,
    page_number: int,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> List[Dict]:
    """Create overlapping word-based chunks while preserving page metadata."""
    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end]).strip()

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "page": page_number,
                }
            )

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def build_chunks(pages: List[Dict]) -> List[Dict]:
    all_chunks = []

    for page in pages:
        all_chunks.extend(
            split_into_chunks(
                page["text"],
                page["page"],
            )
        )

    return all_chunks


# -----------------------------
# Vector index
# -----------------------------
def build_faiss_index(
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
) -> faiss.Index:
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


def retrieve_chunks(
    question: str,
    index: faiss.Index,
    chunks: List[Dict],
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[Tuple[float, Dict]]:
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append((float(score), chunks[int(idx)]))

    return results


# -----------------------------
# Groq answer generation
# -----------------------------
def get_groq_client() -> Groq:
    """Read GROQ_API_KEY from Streamlit secrets first, then environment."""
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Add it in "
            "Streamlit Cloud → App settings → Secrets."
        )

    return Groq(api_key=api_key)


def generate_answer(
    question: str,
    retrieved: List[Tuple[float, Dict]],
) -> str:
    """Generate an answer using only the retrieved policy context."""
    context_parts = []

    for i, (score, chunk) in enumerate(retrieved, start=1):
        context_parts.append(
            f"[Source {i} | PDF page {chunk['page']} | relevance {score:.3f}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n".join(context_parts)
    context = context[:MAX_CONTEXT_CHARS]

    system_prompt = """You are an HR Policy Assistant.

Answer questions using ONLY the provided HR policy context.

Rules:
1. Do not invent policy details.
2. If the answer is not supported by the context, say:
   "I couldn't find that in the uploaded HR policy."
3. Give a clear, professional answer.
4. When possible, cite the relevant PDF page number(s), for example:
   "According to page 4..."
5. If the policy is ambiguous or appears to conflict with itself, explain
   the ambiguity rather than guessing.
6. Do not use outside knowledge to fill missing policy information.
"""

    user_prompt = f"""HR POLICY CONTEXT:

{context}

QUESTION:
{question}

Provide a concise answer grounded in the context above.
"""

    client = get_groq_client()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_completion_tokens=800,
        include_reasoning=False,
    )

    answer = response.choices[0].message.content

    if not answer:
        return "I couldn't generate an answer from the uploaded policy."

    return answer.strip()


# -----------------------------
# Session state helpers
# -----------------------------
def reset_document_state() -> None:
    for key in [
        "pdf_hash",
        "pages",
        "chunks",
        "index",
        "document_name",
        "total_chars",
    ]:
        st.session_state.pop(key, None)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        """
        **1. Upload** an HR policy PDF  
        **2. Extract** text with PyMuPDF  
        **3. Chunk** the policy text  
        **4. Embed** chunks with Sentence Transformers  
        **5. Search** with FAISS  
        **6. Answer** with Groq `openai/gpt-oss-20b`
        """
    )

    st.divider()
    st.subheader("Configuration")
    st.write(f"**LLM:** `{MODEL_NAME}`")
    st.write(f"**Embeddings:** `{EMBEDDING_MODEL}`")
    st.write(f"**Top-K retrieval:** `{TOP_K}`")

    if st.button("🗑️ Clear document", use_container_width=True):
        reset_document_state()
        st.rerun()


# -----------------------------
# Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your HR Policy PDF",
    type=["pdf"],
    help="Upload a text-based HR policy PDF. Scanned image-only PDFs may require OCR.",
)

if uploaded_file is None:
    st.info("👆 Upload an HR Policy PDF to build the knowledge base.")
    st.stop()


# -----------------------------
# Process only when the PDF changes
# -----------------------------
pdf_bytes = uploaded_file.getvalue()
current_hash = hashlib.sha256(pdf_bytes).hexdigest()

if st.session_state.get("pdf_hash") != current_hash:
    reset_document_state()

    with st.status("Building the HR policy knowledge base...", expanded=True) as status:
        st.write("📄 Extracting PDF text...")
        pages = extract_pdf_pages(pdf_bytes)

        if not pages:
            status.update(
                label="No extractable text found",
                state="error",
            )
            st.error(
                "This PDF does not contain extractable text. "
                "If it is a scanned/image-only PDF, OCR is required before using this app."
            )
            st.stop()

        total_chars = sum(len(page["text"]) for page in pages)

        st.write(f"Found text on {len(pages)} page(s).")
        st.write("✂️ Creating overlapping chunks...")
        chunks = build_chunks(pages)

        if not chunks:
            status.update(
                label="No chunks created",
                state="error",
            )
            st.error("No usable text chunks were created from this PDF.")
            st.stop()

        st.write(f"Created {len(chunks)} chunks.")
        st.write("🧠 Creating embeddings and FAISS index...")
        embedding_model = load_embedding_model()
        index = build_faiss_index(chunks, embedding_model)

        st.session_state["pdf_hash"] = current_hash
        st.session_state["pages"] = pages
        st.session_state["chunks"] = chunks
        st.session_state["index"] = index
        st.session_state["document_name"] = uploaded_file.name
        st.session_state["total_chars"] = total_chars

        status.update(
            label="Knowledge base ready",
            state="complete",
        )


# -----------------------------
# Document summary
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("PDF pages", len(st.session_state["pages"]))

with col2:
    st.metric("Text chunks", len(st.session_state["chunks"]))

with col3:
    st.metric("Extracted characters", f"{st.session_state['total_chars']:,}")

st.caption(f"📄 **Current document:** {st.session_state['document_name']}")


# -----------------------------
# Question answering
# -----------------------------
st.subheader("Ask about the policy")

question = st.text_input(
    "Your question",
    placeholder="e.g., How many annual leave days are employees entitled to?",
)

if st.button("🔎 Ask", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    embedding_model = load_embedding_model()

    with st.spinner("Searching the policy and generating an answer..."):
        retrieved = retrieve_chunks(
            question=question.strip(),
            index=st.session_state["index"],
            chunks=st.session_state["chunks"],
            embedding_model=embedding_model,
            top_k=TOP_K,
        )

        try:
            answer = generate_answer(question.strip(), retrieved)
        except Exception as exc:
            st.error(f"Unable to generate the answer: {exc}")
            st.stop()

    st.subheader("Answer")
    st.write(answer)

    with st.expander("🔎 Retrieved policy sections"):
        for i, (score, chunk) in enumerate(retrieved, start=1):
            st.markdown(
                f"**Source {i} — PDF page {chunk['page']} — "
                f"similarity {score:.3f}**"
            )
            st.write(chunk["text"])
            if i < len(retrieved):
                st.divider()


st.divider()
st.caption(
    "⚠️ This assistant retrieves information from the uploaded policy. "
    "It is not a substitute for official HR/legal advice."
)
