import os
from groq import Groq
from tagging2.config import GROQ_MODEL_TAGGING


class LLMClient:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY2")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")

        self.client = Groq(api_key=api_key)
        self.model = GROQ_MODEL_TAGGING

    def call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()