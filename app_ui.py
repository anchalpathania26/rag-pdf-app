import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader

from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

st.set_page_config(layout="wide")
st.title("Chat with Your PDF")

# -------------------------
# Session state
# -------------------------
if "rag" not in st.session_state:   
    st.session_state.rag = None

# -------------------------
# Cached processing (FAST)
# -------------------------
@st.cache_resource
def process_pdf(temp_path):
    loader = PyPDFLoader(temp_path)
    docs = loader.load()

    if not docs:
        return None, "No readable text found"

    store = FaissVectorStore("temp_store")
    store.build_from_documents(docs)

    rag = RAGSearch()
    rag.vectorstore = store

    return rag, None

# -------------------------
# Upload PDF
# -------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.success("PDF uploaded!")

    # Process only once
    if st.session_state.rag is None:
        with st.spinner("Processing PDF... please wait ⏳"):
            rag, error = process_pdf(temp_path)

        if error:
            st.error(error)
            st.stop()

        st.session_state.rag = rag
        st.success("Ready! Ask your question below 👇")

# -------------------------
# Ask Question
# -------------------------
if st.session_state.rag is not None:
    query = st.text_input("Ask a question about your PDF")

    if query:
        with st.spinner("Thinking..."):
            answer = st.session_state.rag.search_and_summarize(query)

        st.write("### Answer:")
        st.write(answer)
else:
    st.info("Upload a PDF to start chatting")