# Clear ROS/system Python imports for every project command.
VENV ?= .venv
PYTHON ?= 3.12
UV ?= uv
PY := PYTHONPATH= $(VENV)/bin/python
PYTEST := PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(VENV)/bin/python -m pytest
HOST ?= 127.0.0.1
PORT ?= 8000

.PHONY: help setup setup-min setup-venv data eda api test test-frontend lint \
        typecheck check export benchmark clean train-pretrain train-finetune \
        train-baseline train-eval docker-build docker-up docker-down

help:
	@echo "make setup          - uv venv + full ML/dev stack (Python 3.11 or 3.12)"
	@echo "make setup-min      - uv venv + all lightweight tests/API/dev tools"
	@echo "                     Override PYTHON=3.11 or VENV=.venv-other explicitly"
	@echo "make api            - local demo/API on 127.0.0.1:8000"
	@echo "make check          - Python tests, frontend tests, lint, typecheck"
	@echo "make data / eda     - download/convert dataset or generate EDA report"
	@echo "make export         - export artifacts using configs/export.yaml"
	@echo "make benchmark      - measure artifacts using configs/export.yaml"
	@echo "make train-pretrain / train-finetune / train-baseline / train-eval"
	@echo "make docker-build / docker-up / docker-down - CPU deployment"
	@echo "make clean          - remove caches (keeps venv + data)"

# uv installs packages directly, including pip when an old venv lacks it.
# An existing venv is checked and reused, never silently replaced.
setup-venv:
	@command -v $(UV) >/dev/null || { echo "Install uv 0.12.19 (https://docs.astral.sh/uv/getting-started/installation/), then retry."; exit 1; }
	@selected=$$(PYTHONPATH= $(UV) python find "$(PYTHON)") || { echo "Install a supported interpreter: uv python install 3.12; then make setup-min PYTHON=3.12"; exit 1; }; \
	PYTHONPATH= "$$selected" -c 'import sys; assert sys.version_info[:2] in {(3, 11), (3, 12)}, "Supported Python versions: 3.11/3.12. Retry with PYTHON=3.12."' || exit 1; \
	if [ -e "$(VENV)" ]; then \
	  [ -x "$(VENV)/bin/python" ] || { echo "$(VENV) exists but has no working Python. Choose VENV=.venv312 or explicitly repair it."; exit 1; }; \
	  expected=$$(PYTHONPATH= "$$selected" -c 'import sys; print(sys.version_info[:2])'); \
	  PYTHONPATH= "$(VENV)/bin/python" -c 'import sys; assert str(sys.version_info[:2]) == sys.argv[1], "Existing venv Python differs from PYTHON. Choose matching PYTHON=3.11/3.12 or a new VENV=.venv312; existing venv was preserved."' "$$expected" || exit 1; \
	else \
	  PYTHONPATH= $(UV) venv --python "$$selected" "$(VENV)" || exit 1; \
	fi
	PYTHONPATH= $(UV) pip install --python "$(VENV)/bin/python" pip==25.3 setuptools==75.6.0

setup: setup-venv
	PYTHONPATH= $(UV) pip install --python "$(VENV)/bin/python" -r requirements-dev.lock.txt
	PYTHONPATH= $(UV) pip install --python "$(VENV)/bin/python" --no-deps --no-build-isolation -e .

setup-min: setup-venv
	PYTHONPATH= $(UV) pip install --python "$(VENV)/bin/python" -r requirements-min.lock.txt
	PYTHONPATH= $(UV) pip install --python "$(VENV)/bin/python" --no-deps --no-build-isolation -e .

data:
	$(PY) -m agridrone.data --config configs/data.yaml

eda:
	$(PY) -m agridrone.eda --config configs/data.yaml

api:
	$(PY) -m uvicorn agridrone.serving:app --host $(HOST) --port $(PORT)

export:
	$(PY) -m agridrone.export --config configs/export.yaml

benchmark:
	$(PY) -m agridrone.benchmark --config configs/export.yaml

train-pretrain:
	$(PY) -m agridrone.train --config configs/train.yaml --stage pretrain

train-finetune:
	$(PY) -m agridrone.train --config configs/train.yaml --stage finetune

train-baseline:
	$(PY) -m agridrone.train --config configs/train.yaml --stage baseline

train-eval:
	$(PY) -m agridrone.train --config configs/train.yaml --stage eval

test:
	$(PYTEST)

test-frontend:
	node --test frontend/tests/*.test.mjs

lint:
	PYTHONPATH= $(VENV)/bin/ruff check src tests api docker

typecheck:
	PYTHONPATH= $(VENV)/bin/mypy

check: test test-frontend lint typecheck

docker-build:
	docker compose -f docker/compose.yaml build

docker-up:
	docker compose -f docker/compose.yaml up --build -d

docker-down:
	docker compose -f docker/compose.yaml down

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find src tests api docker -type d -name __pycache__ -exec rm -rf {} +
