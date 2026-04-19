from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    JSONLoader
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


# ✅ Load from folder (your old logic but improved)
def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from a directory.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")

    documents = []

    # ---------------- PDF ----------------
    for pdf_file in data_path.glob("**/*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] PDF load failed: {pdf_file} -> {e}")

    # ---------------- TXT ----------------
    for txt_file in data_path.glob("**/*.txt"):
        try:
            loader = TextLoader(str(txt_file))
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] TXT load failed: {txt_file} -> {e}")

    # ---------------- CSV ----------------
    for csv_file in data_path.glob("**/*.csv"):
        try:
            loader = CSVLoader(str(csv_file))
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] CSV load failed: {csv_file} -> {e}")

    # ---------------- Excel ----------------
    for xlsx_file in data_path.glob("**/*.xlsx"):
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Excel load failed: {xlsx_file} -> {e}")

    # ---------------- Word ----------------
    for docx_file in data_path.glob("**/*.docx"):
        try:
            loader = Docx2txtLoader(str(docx_file))
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] Word load failed: {docx_file} -> {e}")

    # ---------------- JSON ----------------
    for json_file in data_path.glob("**/*.json"):
        try:
            loader = JSONLoader(
                file_path=str(json_file),
                jq_schema=".",
                text_content=False
            )
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"[ERROR] JSON load failed: {json_file} -> {e}")

    # ✅ Remove empty documents
    documents = [doc for doc in documents if doc.page_content.strip()]

    print(f"[DEBUG] Total valid documents: {len(documents)}")
    return documents


# ✅ NEW: Load from Streamlit uploaded PDF
def load_uploaded_pdf(uploaded_file) -> List[Any]:
    """
    Load PDF from Streamlit file uploader
    """
    from tempfile import NamedTemporaryFile

    documents = []

    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # ✅ Remove empty pages
        documents = [doc for doc in documents if doc.page_content.strip()]

        print(f"[DEBUG] Loaded {len(documents)} valid docs from uploaded PDF")

    except Exception as e:
        print(f"[ERROR] Uploaded PDF load failed: {e}")

    return documents


# ✅ Test locally
if __name__ == "__main__":
    docs = load_all_documents("data")

    print(f"Loaded {len(docs)} documents")

    if docs:
        print("Example doc:")
        print(docs[0].page_content[:300])