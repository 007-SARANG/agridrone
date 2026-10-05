"""Container readiness probe; liveness alone must not mark a missing model healthy."""

import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:8000/ready", timeout=8) as response:
        sys.exit(0 if response.status == 200 else 1)
except (urllib.error.URLError, TimeoutError):
    sys.exit(1)
