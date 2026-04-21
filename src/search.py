import os
import requests
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:
    def __init__(self, persist_dir="faiss_store",
                 embedding_model="all-MiniLM-L6-v2"):

        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        #  API KEY
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Check your secrets.")

        #  MODEL (now flexible — no more breaking)
        self.model_name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

    def call_llm(self, prompt):
        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,  # ✅ dynamic model
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=data)

            print("STATUS CODE:", response.status_code)

            result = response.json()

            print("API RESPONSE:", result)

            if response.status_code != 200:
                return f"HTTP Error {response.status_code}: {result}"

            if "choices" not in result:
                return f"API Error: {result}"

            return result["choices"][0]["message"]["content"]

        except Exception as e:
            return f"Exception occurred: {str(e)}"

    def search_and_summarize(self, query, top_k=5):
        results = self.vectorstore.query(query, top_k=top_k)

        texts = [
            r["metadata"].get("text", "")
            for r in results if r["metadata"]
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
Answer ONLY using the context below.

Question: {query}

Context:
{context}
"""

        return self.call_llm(prompt)