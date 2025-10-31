import os
import io
import tempfile
from datetime import datetime
import streamlit as st

from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag import RAG

# -----------------------------
# Page & minimal theme polish
# -----------------------------
st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide",
    menu_items={
        "About": "RAG-powered document Q&A by Manish Thapa"
    }
)

# Subtle CSS polish (keeps it lightweight)
st.markdown("""
<style>
/* tighten base */
.block-container { padding-top: 1.25rem; }
/* card */
.card { border: 1px solid #e6e8ec; border-radius: 12px; padding: 0.9rem 1rem; background: #fff; }
.card + .card { margin-top: 0.75rem; }
/* soft header */
.h-soft { font-weight: 600; margin-bottom: 0.4rem; }
/* tag */
.tag { display:inline-block; padding: 0.1rem .5rem; border:1px solid #e6e8ec; border-radius:999px; font-size: .75rem; color:#57606a; }
/* author box */
.author {
    background: #f0f2f6; /* light neutral tone */
    color: #222; /* dark readable text */
    border: 1px solid #d0d3d8;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.author a {
    color: #ff4b4b;
    font-weight: 600;
    text-decoration: none;
}
.author a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()

if "processor" not in st.session_state:
    # default chunks; can be updated in Settings
    st.session_state.processor = DocumentProcessor()

if "rag" not in st.session_state:
    st.session_state.rag = RAG(st.session_state.vector_store)

if "uploaded_files" not in st.session_state:
    # [{name, size, chunks}]
    st.session_state.uploaded_files = []

if "chat" not in st.session_state:
    # list of {"role": "user"|"assistant", "content": str, "time": iso}
    st.session_state.chat = []

# -----------------------------
# Helper functions
# -----------------------------
def toast(msg, icon="✅"):
    st.toast(f"{icon} {msg}")

def process_single_file(uploaded_file, processor: DocumentProcessor):
    # persist to temp path for downstream processor
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        chunks = processor.process_document(tmp_path)
        st.session_state.vector_store.add_documents(chunks)
        # de-dup by name
        names = [f["name"] for f in st.session_state.uploaded_files]
        if uploaded_file.name not in names:
            st.session_state.uploaded_files.append({
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "chunks": len(chunks),
            })
        toast(f"Processed {uploaded_file.name} ({len(chunks)} chunks)")
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

def clear_all_docs():
    st.session_state.vector_store = VectorStore()
    st.session_state.uploaded_files = []
    st.session_state.rag = RAG(st.session_state.vector_store)

def remove_doc(name: str):
    """
    Optional: If VectorStore supports deletion by doc-id, call it here.
    For now, we reindex by clearing and re-processing remaining docs in memory.
    """
    remaining = [d for d in st.session_state.uploaded_files if d["name"] != name]
    if len(remaining) == len(st.session_state.uploaded_files):
        toast("Document not found", "ℹ️")
        return
    # Rebuild vector store from remaining files (requires raw files; we only have metadata)
    # Because we don't keep raw bytes, we fallback to a full reset and prompt user to re-upload.
    st.session_state.uploaded_files = remaining
    st.session_state.vector_store = VectorStore()
    st.session_state.rag = RAG(st.session_state.vector_store)
    toast(f"Removed {name}. Please re-upload remaining docs to rebuild the index.", "🗑️")

def export_chat_bytes() -> bytes:
    buff = io.StringIO()
    for m in st.session_state.chat:
        ts = m.get("time", "")
        role = m["role"].capitalize()
        buff.write(f"[{ts}] {role}: {m['content']}\n")
    return buff.getvalue().encode("utf-8")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### 📄 Documents")
    files = st.file_uploader(
        "Upload one or more documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="PDF/TXT/MD supported"
    )

    if files:
        if st.button("Process uploaded files", type="primary", use_container_width=True):
            # progress feedback
            prog = st.progress(0, text="Processing documents...")
            for idx, f in enumerate(files, start=1):
                process_single_file(f, st.session_state.processor)
                prog.progress(idx / len(files), text=f"Processed {idx}/{len(files)}")
            prog.empty()

    st.divider()
    st.markdown("### ⚙️ Settings")
    with st.expander("Advanced chunking", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            chunk_size = st.slider("Chunk size", 200, 3000, value=getattr(st.session_state.processor, "chunk_size", 1000), step=100)
        with c2:
            chunk_overlap = st.slider("Chunk overlap", 0, 600, value=getattr(st.session_state.processor, "chunk_overlap", 200), step=20)

        if st.button("Apply", use_container_width=True):
            st.session_state.processor = DocumentProcessor(chunk_size, chunk_overlap)
            toast("Updated chunking settings")

    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear all docs", use_container_width=True):
            clear_all_docs()
            toast("Cleared all documents", "🧹")
    with colB:
        st.download_button(
            "Export chat (.txt)",
            data=export_chat_bytes(),
            file_name=f"document-assistant-chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt",
            use_container_width=True
        )

    st.divider()
    st.markdown("### 👤 About")
    st.markdown("""
<div class="author">
  <div class="h-soft">Created by Manish Thapa</div>
  <div class="footer">Software Engineer · Backend & Cloud · CS Grad @ SDSU</div>
  <a href="https://www.linkedin.com/in/iammanish041" target="_blank">🔗 Connect on LinkedIn</a>
</div>
""", unsafe_allow_html=True)

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
    # Empty states
    if not st.session_state.uploaded_files:
        st.info("Tip: Upload documents in the sidebar to enable retrieval-augmented answers.")

    # Replay history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Chat input
    user_msg = st.chat_input("Ask about your documents...")
    if user_msg:
        # validate
        if not st.session_state.uploaded_files:
            st.warning("Please upload and process at least one document first.")
        else:
            # append user message
            st.session_state.chat.append({"role": "user", "content": user_msg, "time": datetime.now().isoformat(timespec="seconds")})
            with st.chat_message("user"):
                st.markdown(user_msg)

            # generate answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = st.session_state.rag.generate_answer(user_msg)
                    except Exception as e:
                        st.error(f"Failed to generate answer: {e}")
                        answer = "Sorry—something went wrong while generating an answer."
                st.markdown(answer)

            st.session_state.chat.append({"role": "assistant", "content": answer, "time": datetime.now().isoformat(timespec="seconds")})

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
                            remove_doc(doc["name"])
                            st.rerun()
        else:
            st.info("No documents yet. Use the sidebar to upload.")

    with right:
        st.subheader("Quick actions")
        st.button("Rebuild index", help="Clears index and re-processes uploaded documents (if kept).", disabled=True)
        st.caption("※ Disabled example. For full rebuild, we need original file bytes in memory or storage.")

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
    st.markdown("""
1. **Document Processing** — files are chunked for retrieval  
2. **Vector Embeddings** — chunks → embedding vectors  
3. **Vector Index** — vectors added to FAISS  
4. **Semantic Search** — your question retrieves relevant chunks  
5. **Generation** — LLM composes the final answer using retrieved context  
""")

    # Optional: architecture sketch with Graphviz (kept simple)
    try:
        import graphviz
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR", splines="curved")
        dot.node("A", "📄 Documents")
        dot.node("B", "🔪 Chunking")
        dot.node("C", "🧠 Embeddings")
        dot.node("D", "🗂️ FAISS Index")
        dot.node("E", "🔎 Retrieval")
        dot.node("F", "🤖 LLM Answer")
        dot.edges(["AB","BC","CD","DE","EF"])
        st.graphviz_chart(dot)
    except Exception:
        st.caption("Graphviz not available—skipping diagram.")

# -----------------------------
# Footer
# -----------------------------
st.caption("Pro tip: Use the **Export chat** button to save a transcript.")