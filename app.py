"""RAG PDF Assistant — Streamlit demo application."""

import os
import streamlit as st

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.llm import ask_llm

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PDF = os.path.join(
    os.path.dirname(__file__), "data", "manuel_employe_techcorp.pdf"
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_embedder():
    """Return a cached Embedder instance (loaded once)."""
    return Embedder()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

if "current_pdf_name" not in st.session_state:
    st.session_state.current_pdf_name = "Aucun"

if "question_triggered" not in st.session_state:
    st.session_state.question_triggered = None


# ---------------------------------------------------------------------------
# Helper: rebuild the vector store from a PDF path
# ---------------------------------------------------------------------------
def rebuild_index(pdf_path: str) -> None:
    """Load PDF, chunk, embed and build the FAISS index."""
    embedder = get_embedder()
    pages = load_pdf(pdf_path)
    chunks = chunk_documents(pages)
    vs = VectorStore()
    vs.build(chunks, embedder)
    st.session_state.vector_store = vs


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📄 Document actif")
    st.markdown(f"**{st.session_state.current_pdf_name}**")

    st.markdown("---")
    st.markdown("### 📤 Uploader un PDF")
    uploaded_file = st.file_uploader(
        "Choisissez un fichier PDF",
        type="pdf",
        label_visibility="collapsed",
    )
    st.caption("⚠️ Ne pas uploader de documents confidentiels")

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = os.path.join(
            os.path.dirname(__file__), "data", "uploaded_temp.pdf"
        )
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.current_pdf_name = uploaded_file.name
        rebuild_index(temp_path)
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    if st.button("🔄 Réinitialiser la conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------
st.title("📄 RAG PDF Assistant")
st.markdown("Posez vos questions à partir de vos documents PDF")

# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    st.error(
        "⚠️ **Clé API Mistral non configurée.**\n\n"
        "Pour utiliser l'assistant, créez un fichier `.env` à la racine du "
        "projet avec :\n\n"
        "```\nMISTRAL_API_KEY=votre_cle_api\n```\n\n"
        "Obtenez une clé sur [console.mistral.ai](https://console.mistral.ai)."
    )

# ---------------------------------------------------------------------------
# Default PDF check
# ---------------------------------------------------------------------------
if st.session_state.vector_store is None:
    if os.path.exists(DEFAULT_PDF):
        st.session_state.current_pdf_name = "manuel_employe_techcorp.pdf"
        with st.spinner("Chargement du PDF par défaut…"):
            rebuild_index(DEFAULT_PDF)
        st.rerun()
    else:
        st.info(
            "📖 **Aucun PDF chargé.**\n\n"
            "Le PDF de démonstration (`data/manuel_employe_techcorp.pdf`) "
            "n'est pas encore présent.\n\n"
            "**Pour commencer :**\n"
            "1. Placez un fichier PDF nommé `manuel_employe_techcorp.pdf` "
            "dans le dossier `data/`\n"
            "2. **Ou** uploadez un PDF via la barre latérale\n\n"
            "L'application fonctionnera avec n'importe quel document PDF."
        )

# ---------------------------------------------------------------------------
# Example questions (clickable buttons)
# ---------------------------------------------------------------------------
if st.session_state.vector_store is not None:
    st.markdown("### 💡 Questions exemples")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Combien de jours de congés par an ?", use_container_width=True):
            st.session_state.question_triggered = (
                "Combien de jours de congés par an ?"
            )
    with col2:
        if st.button(
            "Comment soumettre une note de frais ?", use_container_width=True
        ):
            st.session_state.question_triggered = (
                "Comment soumettre une note de frais ?"
            )
    with col3:
        if st.button(
            "Quel est le processus d'onboarding ?", use_container_width=True
        ):
            st.session_state.question_triggered = (
                "Quel est le processus d'onboarding ?"
            )
    st.markdown("---")

# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**Extrait** (page {s['page']}, score: {s['score']:.3f})"
                    )
                    st.markdown(f"> {s['text'][:300]}…")

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
if st.session_state.vector_store is not None:
    # If a question was triggered by a button, use it
    if st.session_state.question_triggered:
        question = st.session_state.question_triggered
        st.session_state.question_triggered = None
        # We'll process it below via rerun logic — but we need to display it first
    else:
        question = st.chat_input("Posez votre question sur le document…")

    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Recherche dans le document…"):
                embedder = get_embedder()
                vs = st.session_state.vector_store
                results = vs.search(question, embedder, k=3)

                if not results:
                    answer = (
                        "Je n'ai pas trouvé cette information dans le document."
                    )
                    sources = []
                else:
                    answer = ask_llm(question, results, api_key)
                    sources = results

            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for s in sources:
                        st.markdown(
                            f"**Extrait** (page {s['page']}, "
                            f"score: {s['score']:.3f})"
                        )
                        st.markdown(f"> {s['text'][:300]}…")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        st.rerun()
