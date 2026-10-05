"""Deployment contracts that must stay true without a real model or Docker."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_compose_is_local_readonly_and_bounded():
    config = yaml.safe_load((ROOT / "docker/compose.yaml").read_text())
    api = config["services"]["api"]
    assert api["ports"] == ["127.0.0.1:8000:8000"]
    assert api["read_only"] is True
    assert api["cpus"] <= 2
    assert api["mem_limit"] == "2g"
    mount = api["volumes"][0]
    assert mount["read_only"] is True
    assert mount["bind"]["create_host_path"] is False
    assert mount["target"] == api["environment"]["AGRIDRONE_MODEL"]


def test_context_allowlist_has_no_private_or_artifact_trees():
    patterns = [
        line for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert patterns[0] == "**"
    # Reject additions that would expose private inputs or artifacts at build time.
    assert set(patterns[1:]) <= {
        "!docker/", "!docker/Dockerfile", "!docker/healthcheck.py",
        "!pyproject.toml", "!README.md", "!requirements-runtime.lock.txt",
        "!src/", "!src/agridrone/", "!src/agridrone/*.py", "!api/", "!api/*.py",
        "!configs/", "!configs/serve.yaml", "!frontend/", "!frontend/index.html",
        "!frontend/app.js", "!frontend/core.mjs", "!frontend/styles.css",
    }
