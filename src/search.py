import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()


class RAGSearch:
    def __init__(self,
                 persist_dir="faiss_store",
                 embedding_model="all-MiniLM-L6-v2",
                 llm_model="llama-3.1-8b-instant"):

        # Create vector store
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        # Load only if exists
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()

        # Load API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found. Check your environment file.")

        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model
        )

        print("LLM initialized:", llm_model)

    def search_and_summarize(self, query, top_k=5):
        results = self.vectorstore.query(query, top_k=top_k)

        texts = []
        for r in results:
            if r["metadata"] and "text" in r["metadata"]:
                texts.append(r["metadata"]["text"])

        context = "\n\n".join(texts)

        if not context.strip():
            return "No relevant information found in the document."

        prompt = f"""
You must answer only using the given context.

If the answer is not present in the context, say:
Not found in context.

Question:
{query}

Context:
{context}

Answer:
"""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return "Error from language model: " + str(e)


if __name__ == "__main__":
    rag = RAGSearch()
    answer = rag.search_and_summarize("What is attention mechanism?", top_k=5)
    print("Answer:", answer)