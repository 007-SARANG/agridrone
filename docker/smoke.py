"""Verify the model-free image contract using only the standard library."""

import http.client
import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(BASE + path, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def wait_for_liveness(timeout: float = 90) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            status, payload = get("/health")
            if status == 200:
                assert isinstance(json.loads(payload), dict)
                break
        # Docker may publish the port before Uvicorn starts accepting HTTP.
        # A reset/disconnect/timeout is transient only during this bounded wait.
        except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException):
            pass
        if time.monotonic() >= deadline:
            raise AssertionError("Container never became live at /health")
        time.sleep(1)


def main() -> None:
    wait_for_liveness()
    assert get("/ready")[0] == 503, "Missing model must never be ready"
    assert get("/")[0] == 200, "Static demo must be packaged"
    assert get("/demo-assets/app.js")[0] == 200, "Demo JavaScript must be packaged"
    print("Model-free image: /health=200, /ready=503, demo/assets=200")


if __name__ == "__main__":
    main()
