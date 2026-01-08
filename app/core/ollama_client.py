import os
import time

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

    timeout_s = float(os.getenv("OLLAMA_TIMEOUT_S", "900"))
    retries = int(os.getenv("OLLAMA_RETRIES", "2"))
    backoff_s = float(os.getenv("OLLAMA_RETRY_BACKOFF_S", "2"))

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        r: requests.Response | None = None
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()
            return (data.get("message") or {}).get("content", "").strip()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
        except requests.HTTPError as e:
            # Retry server-side failures (5xx) a couple of times.
            if r is not None and r.status_code >= 500 and attempt < retries:
                last_exc = requests.HTTPError(f"{e}\nOllama response: {r.text}")
            else:
                raise requests.HTTPError(f"{e}\nOllama response: {r.text if r is not None else ''}") from e

        if attempt < retries:
            time.sleep(backoff_s * (2**attempt))

    # If we get here, we exhausted retries.
    raise last_exc or RuntimeError("Ollama request failed")