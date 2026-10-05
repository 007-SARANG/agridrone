"""Verify the model-free image contract using only the standard library."""

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


def main() -> None:
    deadline = time.monotonic() + 90
    while True:
        try:
            status, payload = get("/health")
            if status == 200:
                assert isinstance(json.loads(payload), dict)
                break
        except urllib.error.URLError:
            pass
        if time.monotonic() >= deadline:
            raise AssertionError("Container never became live at /health")
        time.sleep(1)
    assert get("/ready")[0] == 503, "Missing model must never be ready"
    assert get("/")[0] == 200, "Static demo must be packaged"
    assert get("/demo-assets/app.js")[0] == 200, "Demo JavaScript must be packaged"
    print("Model-free image: /health=200, /ready=503, demo/assets=200")


if __name__ == "__main__":
    main()
