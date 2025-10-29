"""
Prompt templates for Agentic AI — Handles Q&A from documents with intelligent fallback
with memory of previous conversations
"""

from typing import List, Dict

# System prompt for general + context-based QA
SYSTEM_PROMPT_QA = """
You are Agentic AI — an intelligent, polite, and highly knowledgeable assistant.
Your goal is to give clear, accurate, and helpful answers.

Rules:
1. If the provided document context contains the answer, use it as your main source.
2. If the context does not answer the question, rely on your own knowledge to help the user.
3. If the question is unrelated to the document, still provide the best possible answer.
4. Be concise but thorough, avoiding unnecessary filler.
5. Never make up fake sources or page numbers.
6. If the answer involves a list, format it neatly with bullet points or numbers.
7. If the question is ambiguous, ask clarifying questions before answering.
8. Always maintain a professional and friendly tone.
9. Support any type of document (PDF, TXT, DOCX, PPTX, CSV, JSON, XML, etc.).
10. Remember previous conversation history for context-aware answers.
"""

def get_qa_prompt(
    context: str,
    question: str,
    history: List[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Generate a QA prompt that works with any file context + general knowledge,
    while remembering conversation history.
    
    Parameters:
        context (str): Extracted relevant text from the document (may be empty).
        question (str): User's question.
        history (List[Dict[str, str]], optional): Previous conversation history.

    Returns:
        List[Dict[str, str]]: Chat format prompt for the LLM.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT_QA.strip()}]

    # Add past conversation if available
    if history:
        messages.extend(history)

    # Add current question with context
    messages.append(
        {"role": "user", "content": f"Document Context:\n{context}\n\nUser Question: {question}"}
    )

    return messages
