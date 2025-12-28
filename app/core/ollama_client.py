import requests
from app.core.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def ollama_chat(prompt: str, *, system: str = "", temperature: float = 0.2) -> str:
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": ([{"role": "system", "content": system}] if system else [])
        + [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=300)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # Print Ollama's actual error body to your terminal
        raise requests.HTTPError(f"{e}\nOllama response: {r.text}") from e

    data = r.json()
    return (data.get("message") or {}).get("content", "").strip()