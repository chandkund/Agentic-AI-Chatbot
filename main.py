"""
Agentic AI Backend Service - PDF + LLM Fallback
"""

import os
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ChromaDB imports
import chromadb
from chromadb.config import Settings

# Local imports
from openrouter_llm import OpenRouterLLM

# Configuration
load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("agentic-ai")

# Constants
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "agentic")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 30))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

# Global state
db = None
retriever = None
qa_chain = None
llm = None


def clean_chroma_db():
    """Completely clean up ChromaDB directory and reset state"""
    try:
        if hasattr(chromadb, "_instance"):
            del chromadb._instance
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        os.makedirs(PERSIST_DIR, exist_ok=True)
    except Exception as e:
        logger.error(f"Error cleaning ChromaDB: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, retriever, qa_chain, llm
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    clean_chroma_db()

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        db = initialize_chroma(embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 6})

        llm = OpenRouterLLM(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, return_source_documents=True
        )

        yield
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        raise


def initialize_chroma(embeddings) -> Chroma:
    max_retries = 3
    last_exception = None

    for attempt in range(max_retries):
        try:
            if hasattr(chromadb, "_instance"):
                del chromadb._instance

            client = chromadb.PersistentClient(
                path=PERSIST_DIR,
                settings=Settings(
                    anonymized_telemetry=False, allow_reset=True, is_persistent=True
                ),
            )

            client.get_or_create_collection(
                name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )

            return Chroma(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=PERSIST_DIR,
            )

        except Exception as e:
            last_exception = e
            logger.warning(f"Chroma init attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                clean_chroma_db()

    raise RuntimeError(
        f"Failed to initialize ChromaDB after {max_retries} attempts"
    ) from last_exception


# FastAPI App
app = FastAPI(title="Agentic AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class QueryIn(BaseModel):
    question: str = Field(..., min_length=3)


# Helper Functions
def safe_filename(filename: str) -> str:
    return Path(filename).name


def loader_for_file(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(path)
    if ext in (".docx", ".doc"):
        return Docx2txtLoader(path)
    if ext in (".txt", ".md"):
        return TextLoader(path)
    raise ValueError(f"Unsupported file type: {ext}")


# API Endpoints
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chroma_status": "ready" if db else "uninitialized",
        "llm_status": "ready" if llm else "uninitialized",
    }




# --- Add at the top ---
from collections import defaultdict

# Global conversation memory (per user_id)
conversation_memory = defaultdict(list)


def build_prompt_with_memory(user_id: str, question: str, context: Optional[str] = ""):
    """Combine memory + new question into a single prompt"""
    history = conversation_memory[user_id]

    messages = [
        {"role": "system", "content": "You are Agentic AI — a helpful assistant with memory."}
    ]

    # Add conversation history
    messages.extend(history)

    # Add new message
    if context:
        user_msg = f"Document Context:\n{context}\n\nUser Question: {question}"
    else:
        user_msg = question

    messages.append({"role": "user", "content": user_msg})
    return messages





@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    tmp_path = None
    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(400, f"File exceeds {MAX_UPLOAD_MB}MB limit")

        filename = safe_filename(file.filename)
        tmp_path = os.path.join(UPLOAD_DIR, f"tmp_{filename}")
        with open(tmp_path, "wb") as f:
            f.write(contents)

        loader = loader_for_file(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        db.add_documents(chunks)
        db.persist()

        return {"message": f"Uploaded {filename} ({len(chunks)} chunks)"}

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# @app.post("/query")
# async def query_document(query: QueryIn):
#     try:
#         # Step 1: Try searching in PDF (ChromaDB)
#         result = qa_chain({"query": query.question})
#         answer = result.get("result", "").strip()

#         # Step 2: If not found, fallback to LLM
#         if not answer or answer.lower() in [
#             "i don't know.",
#             "i am not sure.",
#             "",
#         ]:
#             logger.info("Falling back to OpenRouter LLM.")
#             answer = llm(f"Answer the following question:\n\n{query.question}")

#         return {"answer": answer}

#     except Exception as e:
#         logger.error(f"Query failed: {e}", exc_info=True)
#         raise HTTPException(500, str(e))

@app.post("/query")
async def query_document(query: QueryIn, user_id: str = "default"):
    try:
        # Step 1: Try searching in ChromaDB
        result = qa_chain({"query": query.question})
        answer = result.get("result", "").strip()

        # Step 2: Fallback to raw LLM if no answer
        if not answer or answer.lower() in ["i don't know.", "i am not sure.", ""]:
            logger.info("Falling back to OpenRouter LLM.")

            messages = build_prompt_with_memory(user_id, query.question)
            answer = llm(messages)

        # Save conversation to memory
        conversation_memory[user_id].append({"role": "user", "content": query.question})
        conversation_memory[user_id].append({"role": "assistant", "content": answer})

        return {"answer": answer}

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))
    


@app.post("/reset_memory")
async def reset_memory(user_id: str = "default"):
    conversation_memory[user_id] = []
    return {"status": f"Memory cleared for user {user_id}"}





@app.post("/reset")
async def reset_db():
    try:
        global db, retriever, qa_chain
        clean_chroma_db()
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        db = initialize_chroma(embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 6})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, return_source_documents=True
        )
        return {"status": "Database reset successfully"}
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
