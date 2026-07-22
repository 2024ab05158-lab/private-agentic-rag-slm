from config import SLM_MODEL

import requests

from application.utils.process_monitor import ProcessMonitor


def generate_response(prompt):
    """
    Sends prompt to the locally running
    Small Language Model using Ollama.

    CPU monitoring starts immediately before
    inference and stops immediately after
    the response is received.
    """

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": SLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    # ------------------------------------
    # Start CPU Monitoring
    # ------------------------------------

    ProcessMonitor.start()

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=300
        )

    finally:

        # Always stop monitoring
        ProcessMonitor.stop()

    if response.status_code == 200:

        return response.json()["response"]

    return "Unable to generate response from local SLM."