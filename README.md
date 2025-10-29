# 🤖 Agentic AI Chatbot – Chat with Your Own Documents

Agentic AI Chatbot is an intelligent conversational assistant that enables users to **interact with their own documents** — including PDFs, DOCX, TXT, and Markdown files — through **Large Language Models (LLMs)** integrated with **LangChain** and **ChromaDB**.
It supports **context-aware conversation memory**, **vector-based retrieval**, and **OpenRouter-powered reasoning**, providing accurate and human-like answers from your private data.

---

## 🚀 Features

* 📂 **Multi-file support:** PDF, DOCX, TXT, Markdown
* 🧠 **Contextual memory:** remembers prior chat context
* 🔍 **Intelligent search:** ChromaDB vector similarity search
* 🧩 **LLM integration:** OpenRouter API fallback for reasoning
* ⚡ **FastAPI backend** + **Streamlit frontend**
* 🐳 **Docker-ready:** seamless containerized deployment

---

## 🧱 Project Structure

```
Agentic-Chatbot/
│
├── main.py                # FastAPI backend logic
├── prompt.py              # Prompt templates and memory logic
├── openrouter_llm.py      # OpenRouter LLM API integration
├── streamlit_app.py       # Streamlit frontend UI
│
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Compose file for simplified deployment
├── .dockerignore          # Files excluded from image build
│
├── uploads/               # Temporary upload directory
├── chroma_db/             # Persistent vector database
└── .env                   # Environment variables
```

---

## ⚙️ Installation (Local Setup)

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/agentic-ai-chatbot.git
   cd agentic-ai-chatbot
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file in the project root:

   ```
   OPENROUTER_API_KEY=your_openrouter_api_key
   CHROMA_PERSIST_DIR=./chroma_db
   UPLOAD_DIR=./uploads
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```

5. **Run the backend**

   ```bash
   uvicorn main:app --reload --port 8000
   ```

6. **Run the frontend**

   ```bash
   streamlit run streamlit_app.py
   ```

---

## 🐳 Running with Docker

1️⃣ **Build the Docker image**

```bash
docker build -t agentic-ai-chatbot .
```

2️⃣ **Run the container**

```bash
docker run -p 8501:8501 agentic-ai-chatbot
```

3️⃣ **Or use Docker Compose**

```bash
docker-compose up --build
```

---

## 💬 API Endpoints

| Endpoint        | Method | Description                         |
| --------------- | ------ | ----------------------------------- |
| `/upload`       | POST   | Upload a document (PDF/DOCX/TXT/MD) |
| `/query`        | POST   | Query uploaded documents            |
| `/reset`        | POST   | Reset Chroma database               |
| `/reset_memory` | POST   | Clear session memory                |
| `/health`       | GET    | Check API health and LLM connection |

---

## 🧠 How It Works

1. Upload your documents via the Streamlit interface.
2. The files are split into chunks and embedded using **HuggingFace sentence transformers**.
3. These embeddings are stored in **ChromaDB** for vector similarity search.
4. When you ask a question:

   * It first searches the vector store for relevant chunks.
   * If no match is found, it uses **OpenRouter’s LLM** for fallback reasoning.
5. The conversation context is maintained for more natural interactions.

---

## 🧩 Environment Variables

| Variable             | Description                    | Default            |
| -------------------- | ------------------------------ | ------------------ |
| `OPENROUTER_API_KEY` | API key for OpenRouter         | Required           |
| `OPENROUTER_MODEL`   | LLM model to use               | openai/gpt-4o-mini |
| `CHROMA_PERSIST_DIR` | Directory for ChromaDB storage | ./chroma_db        |
| `UPLOAD_DIR`         | Directory for uploaded files   | ./uploads          |
| `MAX_UPLOAD_MB`      | Max file upload size           | 30                 |

---

## 🧰 Tech Stack

* **Backend:** FastAPI, LangChain
* **Frontend:** Streamlit
* **Database:** ChromaDB
* **Embeddings:** Sentence Transformers
* **LLM API:** OpenRouter
* **Containerization:** Docker + Docker Compose

---

## 👨‍💻 Author

**Chandan Kumar**
🎓 Data Science Graduate | 💡 AI Developer | 📊 ML Enthusiast
🔗 [GitHub](https://github.com/chandkund) | [LinkedIn](https://linkedin.com/in/chandankund)

---

## 🪪 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute it with proper attribution.

---

## 🌟 Acknowledgments

* [LangChain](https://www.langchain.com/)
* [ChromaDB](https://www.trychroma.com/)
* [Sentence Transformers](https://www.sbert.net/)
* [OpenRouter](https://openrouter.ai/)
