# AgriDrone developer commands.
#
# NOTE: this project unsets PYTHONPATH for every Python invocation. The dev
# machine sources a ROS 2 workspace that exports PYTHONPATH, which otherwise
# leaks system site-packages (and a broken pytest plugin) into the venv.

VENV := .venv
PY := PYTHONPATH= $(VENV)/bin/python
PYTEST := PYTHONPATH= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(VENV)/bin/python -m pytest

.PHONY: help setup setup-min data eda test lint typecheck clean \
        train-pretrain train-finetune train-baseline train-eval

help:
	@echo "make setup          - create venv + install all deps (needs internet)"
	@echo "make setup-min      - venv + lightweight data-pipeline deps only (no torch)"
	@echo "make data           - download + convert PlantDoc to YOLO format"
	@echo "make eda            - generate the EDA report + plots"
	@echo "make test           - run the test suite"
	@echo "make lint           - ruff check"
	@echo "make typecheck      - mypy"
	@echo "make train-pretrain - PlantVillage backbone pretrain (needs GPU + full stack)"
	@echo "make train-finetune - PlantDoc detector fine-tune  (needs GPU + full stack)"
	@echo "make train-baseline - YOLOv8 baseline fine-tune     (needs GPU + full stack)"
	@echo "make train-eval     - evaluate on TEST -> results table (needs full stack)"
	@echo "make clean          - remove caches (keeps venv + data)"

# Full setup. If your system lacks pip in venvs (Debian/PEP 668), see README
# 'Environment setup' for the get-pip.py bootstrap.
setup:
	python3 -m venv $(VENV) || python3 -m venv --without-pip $(VENV)
	$(PY) -m ensurepip --upgrade 2>/dev/null || true
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

# Lightweight: enough to run the data pipeline + tests without the ML stack.
setup-min:
	$(PY) -m pip install pandas numpy Pillow matplotlib PyYAML pytest
	$(PY) -m pip install -e . --no-deps

data:
	$(PY) -m agridrone.data --config configs/data.yaml

eda:
	$(PY) -m agridrone.eda --config configs/data.yaml

# Training stages. These need a GPU and the full ML stack (`make setup`); they are
# normally run on Colab via notebooks/phase2_colab.ipynb, not on the dev machine.
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

lint:
	PYTHONPATH= $(VENV)/bin/ruff check src tests

typecheck:
	PYTHONPATH= $(VENV)/bin/mypy

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
