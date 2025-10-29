# streamlit_llm.py
import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 30))

st.set_page_config(page_title="Agentic AI", layout="wide")
st.title("📄 Agentic AI — Your Document Assistant")

def call_api(endpoint, method="POST", json=None, files=None, timeout=120):
    url = f"{BACKEND_URL}{endpoint}"
    try:
        if method.upper() == "POST":
            if files:
                resp = requests.post(url, files=files, timeout=timeout)
            else:
                resp = requests.post(url, json=json, timeout=timeout)
        else:
            resp = requests.get(url, params=json, timeout=timeout)
        if not resp.ok:
            st.error(f"API Error {resp.status_code}: {resp.text}")
            return None
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

# Only two tabs now: Upload & Ask
tabs = st.tabs(["📤 Upload", "❓ Ask"])
tab1, tab2 = tabs

# ----------------- TAB 1: Upload -----------------
with tab1:
    st.subheader("Upload a Document")
    uploaded_file = st.file_uploader("Choose file", type=["pdf", "docx", "txt"])
    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.write(f"{uploaded_file.name} — {file_size_mb:.2f} MB")
        if file_size_mb > MAX_UPLOAD_MB:
            st.warning(f"File exceeds {MAX_UPLOAD_MB} MB limit.")
        else:
            if st.button("Upload File"):
                with st.spinner("Uploading and ingesting..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    res = call_api("/upload", files=files)
                    if res:
                        st.success(res.get("message", "Upload successful"))

# ----------------- TAB 2: Chat Ask -----------------
with tab2:
    st.subheader("Ask a Question (Chat Mode)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # (role, text)

    # Display past conversation
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    # Input for new question
    q = st.chat_input("Type your question here...")
    if q:
        st.session_state.chat_history.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)

        with st.spinner("Thinking..."):
            res = call_api("/query", json={"question": q})

        if res:
            answer = res.get("answer", "No answer found.")
            st.session_state.chat_history.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(answer)
