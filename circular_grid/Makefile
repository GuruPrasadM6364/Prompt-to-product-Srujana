# EcoFlow Campus - Makefile
# One-command reproducible pipeline. All targets are phased.
# Usage: make setup | make data | make train | make test | make run

PYTHON = python
PROJECT = .
FRONTEND = ./frontend

.PHONY: all setup data demo-data train test run clean

all: setup data train test

## ── 1. Setup: install Python packages + Node packages ─────────────────────────
setup:
	@echo "=== Installing Python dependencies ==="
	$(PYTHON) -m pip install -r requirements.txt
	@echo "=== Setting up Node/frontend ==="
	cd $(FRONTEND) && npm install
	@echo "=== Building frontend ==="
	cd $(FRONTEND) && npm run build
	@echo "Setup complete."

## ── 2. Data: audit and convert the supplied source workbooks ─────────────────
data:
	@echo "=== Auditing supplied source workbooks ==="
	cd $(PROJECT) && $(PYTHON) -c "from src.ingestion.audit import run_audit; run_audit()"
	@echo "=== Data audit complete. Reports in data/ ==="

## ── Explicitly synthetic data for local demos only ──────────────────────────
demo-data:
	@echo "=== Generating synthetic waste events for demo use only ==="
	cd $(PROJECT) && $(PYTHON) scripts/generate_waste_events.py
	@echo "Synthetic demo data generated; do not use it as measured source data."

## ── 3. Train: classifier + forecasting models ────────────────────────────────
train:
	@echo "=== Phase 3: Training waste classifier ==="
	cd $(PROJECT) && $(PYTHON) -c "from src.classifier.train import run_classifier; run_classifier()"
	@echo "=== Phase 4: Training forecasting models ==="
	cd $(PROJECT) && $(PYTHON) -c "from src.forecasting.forecast import run_forecasting; run_forecasting()"
	@echo "=== Train phase complete. Models in models/ ==="

## ── 4. Test: run full pytest suite ───────────────────────────────────────────
test:
	@echo "=== Running test suite ==="
	cd $(PROJECT) && $(PYTHON) -m pytest tests/ -v --tb=short
	@echo "=== Tests complete ==="

## ── 5. Run: start the application server ────────────────────────────────────
run:
	@echo "=== Starting EcoFlow Campus server on http://localhost:8000 ==="
	cd $(PROJECT) && $(PYTHON) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

## ── Scale test ───────────────────────────────────────────────────────────────
scale-test:
	@echo "=== Running scale load test (8 → 32 buildings) ==="
	cd $(PROJECT) && $(PYTHON) scripts/scale_test.py

## ── Clean generated files (keeps originals) ──────────────────────────────────
clean:
	@echo "=== Cleaning generated files ==="
	cd $(PROJECT) && rm -f data/*.csv data/*.json models/*.pkl models/*.onnx models/*.json
	@echo "Clean complete."
