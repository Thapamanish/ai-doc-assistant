import os
import io
import tempfile
from datetime import datetime
import streamlit as st

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag import RAG

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide",
    menu_items={"About": "RAG-powered document Q&A by Manish Thapa"},
)

# -----------------------------
# Session state initialization
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()

if "rag" not in st.session_state:
    st.session_state.rag = RAG(st.session_state.vector_store)

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "chat" not in st.session_state:
    st.session_state.chat = []

# -----------------------------
# Helpers
# -----------------------------
def toast(msg, icon="✅"):
    fn = getattr(st, "toast", None)
    if callable(fn):
        fn(f"{icon} {msg}")

def _process_uploaded_file_core(uploaded_file, processor: DocumentProcessor):
    """
    Core logic to persist an uploaded file to a temp path, process it,
    add chunks to the vector store, update session metadata, and clean up.
    This function uses tempfile.NamedTemporaryFile so tests can patch it.
    """
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        chunks = processor.process_document(tmp_path)
        st.session_state.vector_store.add_documents(chunks)

        # de-dup by name
        names = [f["name"] for f in st.session_state.uploaded_files]
        if uploaded_file.name not in names:
            st.session_state.uploaded_files.append(
                {"name": uploaded_file.name, "size": uploaded_file.size, "chunks": len(chunks)}
            )

        toast(f"Processed {uploaded_file.name} ({len(chunks)} chunks)")
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def process_single_file(uploaded_file, processor: DocumentProcessor):
    """
    UI helper: call the core routine with the provided processor.
    (Kept for your sidebar flow.)
    """
    return _process_uploaded_file_core(uploaded_file, processor)

def clear_all_docs():
    """Reset all documents and RAG components."""
    st.session_state.vector_store = VectorStore()
    st.session_state.uploaded_files = []
    st.session_state.rag = RAG(st.session_state.vector_store)

def export_chat_bytes() -> bytes:
    buff = io.StringIO()
    for m in st.session_state.chat:
        ts = m.get("time", "")
        role = m["role"].capitalize()
        buff.write(f"[{ts}] {role}: {m['content']}\n")
    return buff.getvalue().encode("utf-8")

# -----------------------------
# Wrappers for tests (public API)
# -----------------------------
def process_document(uploaded_file):
    """
    Public wrapper used by tests.
    IMPORTANT: This function calls tempfile.NamedTemporaryFile directly,
    so the test's patch('tempfile.NamedTemporaryFile') observes the call.
    """
    return _process_uploaded_file_core(uploaded_file, st.session_state.processor)

def clear_documents():
    """Public wrapper for tests."""
    return clear_all_docs()

# Dynamic attribute for test access (e.g., app.processor)
def __getattr__(name):
    if name == "processor":
        return st.session_state.processor
    raise AttributeError(f"module 'app' has no attribute {name!r}")

# -----------------------------
# Sidebar UI
# -----------------------------
with st.sidebar:
    st.markdown("### 📄 Documents")
    files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if files:
        if st.button("Process uploaded files", type="primary", use_container_width=True):
            prog = st.progress(0, text="Processing documents...")
            for i, f in enumerate(files, start=1):
                process_single_file(f, st.session_state.processor)
                prog.progress(i / len(files), text=f"Processed {i}/{len(files)}")
            prog.empty()

    st.divider()
    st.markdown("### ⚙️ Settings")
    with st.expander("Advanced chunking", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            chunk_size = st.slider(
                "Chunk size",
                200,
                3000,
                value=getattr(st.session_state.processor, "chunk_size", 1000),
                step=100,
            )
        with c2:
            chunk_overlap = st.slider(
                "Chunk overlap",
                0,
                600,
                value=getattr(st.session_state.processor, "chunk_overlap", 200),
                step=20,
            )

        if st.button("Apply", use_container_width=True):
            st.session_state.processor = DocumentProcessor(chunk_size, chunk_overlap)
            toast("Updated chunking settings")

    if st.button("Clear all docs", use_container_width=True):
        clear_all_docs()
        toast("Cleared all documents", "🧹")

    st.download_button(
        "Export chat (.txt)",
        data=export_chat_bytes(),
        file_name=f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt",
        use_container_width=True,
    )

# -----------------------------
# Header
# -----------------------------
st.title("🤖 Document Assistant")
st.caption("Upload documents and ask questions about their contents using RAG.")

# -----------------------------
# Main layout (Tabs)
# -----------------------------
tab_chat, tab_docs, tab_insights = st.tabs(["💬 Chat", "📚 Documents", "📊 Insights"])

# -----------------------------
# TAB: Chat
# -----------------------------
with tab_chat:
    if not st.session_state.uploaded_files:
        st.info("Tip: Upload documents in the sidebar to enable retrieval-augmented answers.")

    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Ask about your documents...")
    if user_msg:
        if not st.session_state.uploaded_files:
            st.warning("Please upload and process at least one document first.")
        else:
            st.session_state.chat.append(
                {"role": "user", "content": user_msg, "time": datetime.now().isoformat(timespec="seconds")}
            )
            with st.chat_message("user"):
                st.markdown(user_msg)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = st.session_state.rag.generate_answer(user_msg)
                    except Exception as e:
                        st.error(f"Failed to generate answer: {e}")
                        answer = "Sorry—something went wrong while generating an answer."
                st.markdown(answer)

            st.session_state.chat.append(
                {"role": "assistant", "content": answer, "time": datetime.now().isoformat(timespec="seconds")}
            )

# -----------------------------
# TAB: Documents
# -----------------------------
with tab_docs:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Uploaded")
        if st.session_state.uploaded_files:
            for doc in st.session_state.uploaded_files:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([6, 3, 3])
                    with c1:
                        st.markdown(f"**{doc['name']}**")
                        st.caption(f"{doc['size']/1024:.1f} KB · {doc['chunks']} chunks")
                    with c2:
                        st.markdown('<span class="tag">Indexed</span>', unsafe_allow_html=True)
                    with c3:
                        if st.button("Remove", key=f"rm-{doc['name']}"):
                            # Simple rebuild path; keeping consistent with earlier helper
                            st.session_state.uploaded_files = [
                                d for d in st.session_state.uploaded_files if d["name"] != doc["name"]
                            ]
                            st.session_state.vector_store = VectorStore()
                            st.session_state.rag = RAG(st.session_state.vector_store)
                            toast(f"Removed {doc['name']}. Please re-upload to rebuild index.", "🗑️")
                            st.rerun()
        else:
            st.info("No documents yet. Use the sidebar to upload.")

    with right:
        st.subheader("Quick actions")
        st.button("Rebuild index", help="Would rebuild from raw files if stored.", disabled=True)
        st.caption("※ Disabled example. For a full rebuild, we would need original file bytes in storage.")

# -----------------------------
# TAB: Insights
# -----------------------------
with tab_insights:
    st.subheader("System Statistics")
    docs_count = len(st.session_state.uploaded_files)
    total_chunks = sum(d["chunks"] for d in st.session_state.uploaded_files) if docs_count else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Documents", docs_count)
    with m2:
        st.metric("Total Chunks", total_chunks)
    with m3:
        st.metric("Chat Turns", len(st.session_state.chat))

    st.markdown("#### How it works")
    st.markdown(
        """
1. **Document Processing** — files are chunked for retrieval  
2. **Vector Embeddings** — chunks → embedding vectors  
3. **Vector Index** — vectors added to FAISS  
4. **Semantic Search** — your question retrieves relevant chunks  
5. **Generation** — LLM composes the final answer using retrieved context  
"""
    )

# -----------------------------
# Footer
# -----------------------------
st.caption("Pro tip: Use the **Export chat** button to save a transcript.")