# 📄 Chat with Your PDF (RAG Application)

🚀 An end-to-end AI-powered application that allows users to upload PDF documents and interact with them using natural language queries.

🔗 **Live App:** https://rag-pdf-app-9rkxbkvbjvbe9gdrxkmfwl.streamlit.app

---

## 📸 Demo

![App Screenshot](<img width="1154" height="649" alt="image" src="https://github.com/user-attachments/assets/fecd787e-1354-49a3-a4a2-9dd055af042f" />
)

> Upload a PDF → Ask questions → Get intelligent answers instantly

---

## ✨ What This Project Does

This application enables users to:

* 📤 Upload any PDF document
* 🔍 Extract and process text automatically
* ❓ Ask questions in natural language
* 🤖 Receive accurate, context-based answers powered by AI

---

## 🧠 How It Works (RAG Pipeline)

This project is built using a **Retrieval-Augmented Generation (RAG)** approach:

1. **Document Loading**

   * Extracts text from uploaded PDFs

2. **Chunking**

   * Splits text into smaller meaningful pieces

3. **Embedding Generation**

   * Converts text into vectors using `Sentence Transformers`

4. **Vector Storage**

   * Stores embeddings in a **FAISS vector database**

5. **Query Processing**

   * Finds the most relevant chunks based on user query

6. **LLM Response Generation**

   * Sends context to **Groq LLM** to generate accurate answers

---

## ⚙️ Tech Stack

* **Python**
* **Streamlit** – Interactive UI
* **FAISS** – Vector Database
* **Sentence Transformers** – Embeddings
* **Groq API (LLM)** – Response generation
* **LangChain** – Document processing

---

## 📂 Project Structure

```
├── src/
│   ├── data_loader.py
│   ├── embedding.py
│   ├── vectorstore.py
│   ├── search.py
├── app_ui.py
├── requirements.txt
├── README.md
```

---

## ▶️ How to Use

1. Open the **Live App**
2. Upload a PDF file
3. Ask a question like:

   * “Summarize this document”
   * “What are the key points?”
4. Get instant AI-generated answers

---

## 🎯 Example Use Cases

* 📄 Resume analysis
* 📚 Research paper summarization
* 🧾 Document Q&A systems
* 🧠 Knowledge base assistant

---

## 💡 Key Highlights

* Built a **complete RAG pipeline from scratch**
* Integrated **vector search with LLM reasoning**
* Developed a **fully interactive web app**
* Deployed and made publicly accessible

---

## 🚀 Future Improvements

* Chat history (like ChatGPT)
* Multiple PDF support
* Model selection from UI
* Faster retrieval with caching

---

## 👨‍💻 Author

**Anchal Pathania**
