# openrouter_llm.py
"""
LangChain-compatible OpenRouter LLM wrapper
"""

import os
import requests
from typing import Optional, List, Dict, Any
from langchain.llms.base import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from dotenv import load_dotenv

load_dotenv()

class OpenRouterLLM(LLM):
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 800
    api_key: str = os.getenv("OPENROUTER_API_KEY")
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise ValueError(f"OpenRouter API request failed: {e}")