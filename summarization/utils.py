import json
import re


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(uh|um|basically|actually|you know)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
