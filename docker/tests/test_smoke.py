"""Startup retries must tolerate transport races, not hide broken contracts."""

import http.client
import importlib.util
import urllib.error
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "agridrone_docker_smoke", Path(__file__).resolve().parents[1] / "smoke.py"
)
assert spec is not None and spec.loader is not None
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.mark.parametrize("error", [
    ConnectionResetError("Connection reset by peer"),
    ConnectionAbortedError("Connection aborted"),
    TimeoutError("Timed out"),
    http.client.RemoteDisconnected("Closed before response"),
    urllib.error.URLError("Connection refused"),
])
def test_transient_startup_failure_then_full_contract_passes(monkeypatch, error):
    calls = []
    sleeps = []

    def get(path):
        calls.append(path)
        if len(calls) == 1:
            raise error
        return (503, b'{}') if path == "/ready" else (200, b'{}')

    monkeypatch.setattr(smoke, "get", get)
    monkeypatch.setattr(smoke.time, "sleep", sleeps.append)
    smoke.main()
    assert calls == ["/health", "/health", "/ready", "/", "/demo-assets/app.js"]
    assert sleeps == [1]


def test_persistent_connection_reset_fails_at_deadline(monkeypatch):
    times = iter([0, 0, 1, 2])
    monkeypatch.setattr(smoke.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(smoke.time, "sleep", lambda _: None)

    def fail(_):
        raise ConnectionResetError("Still unavailable")

    monkeypatch.setattr(smoke, "get", fail)
    with pytest.raises(AssertionError, match="Container never became live"):
        smoke.wait_for_liveness(timeout=2)


def test_invalid_liveness_payload_is_not_retried(monkeypatch):
    monkeypatch.setattr(smoke, "get", lambda _: (200, b'[]'))
    with pytest.raises(AssertionError):
        smoke.wait_for_liveness()


def test_missing_model_readiness_must_still_fail(monkeypatch):
    monkeypatch.setattr(smoke, "get", lambda _: (200, b'{}'))
    with pytest.raises(AssertionError, match="Missing model must never be ready"):
        smoke.main()
