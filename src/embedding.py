from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)

        print(f"[INFO] Loaded embedding model: {model_name}")

    # ✅ Split documents into chunks
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        if not documents:
            print("❌ No documents to split")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        # ✅ Remove empty chunks
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} valid chunks.")

        return chunks

    # ✅ Convert chunks into embeddings
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        if not chunks:
            print("❌ No chunks to embed")
            return np.array([])

        # ✅ Remove empty text
        texts = [chunk.page_content.strip() for chunk in chunks if chunk.page_content.strip()]

        if not texts:
            print("❌ No valid text found in chunks")
            return np.array([])

        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(texts, show_progress_bar=True)

        print(f"[INFO] Embeddings shape: {embeddings.shape}")

        return embeddings


# ✅ Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")

    emb_pipe = EmbeddingPipeline()

    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)

    if len(embeddings) > 0:
        print("[INFO] Example embedding:", embeddings[0])
    else:
        print("❌ No embeddings generated")