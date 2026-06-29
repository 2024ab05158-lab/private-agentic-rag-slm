from config import SLM_MODEL

import requests


def generate_response(prompt):
    """
    Sends prompt to locally running
    Small Language Model using Ollama.
    """

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": SLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()["response"]

    return "Unable to generate response from local SLM."