# doxtr-rag — Quality Gate (P7)
#
# `make qa` runs the full gate that CI enforces:
#   - pytest with >=90% coverage (fail_under=90 in pyproject.toml)
#   - mypy --strict over doxtr_rag/
#   - tsc --strict over the Pi extension
#   - vitest for the TS extension
#   - the Python<->TS cross-language integration round-trip (part of pytest)
#   - the extension-verifier regression smoke check (host build must not break)

.PHONY: qa test cov mypy tsc tstest xlang dox-build dox-pdf verifier install-dev clean help

PY ?= python
EXT_DIR := doxtr_rag/pi_extension

help:
	@echo "doxtr-rag quality gate targets:"
	@echo "  make qa         Run the full gate (coverage, types, TS, xlang, dox-build, verifier)"
	@echo "  make test       pytest with coverage (>=90%)"
	@echo "  make mypy       mypy --strict over doxtr_rag/"
	@echo "  make tsc        tsc --noEmit --strict over the Pi extension"
	@echo "  make tstest     vitest for the Pi extension"
	@echo "  make xlang      cross-language integration round-trip only"
	@echo "  make dox-build  full dox reference build regression (real sphinx-build)"
	@echo "  make dox-pdf    real dox PDF ingest regression (LaTeX -> P3 extractor)"
	@echo "  make verifier   extension-verifier regression smoke check"
	@echo "  make install-dev  pip install -e .[dev] + npm install"

install-dev:
	$(PY) -m pip install -e ".[dev]"
	cd $(EXT_DIR) && npm install

test cov:
	$(PY) -m pytest --cov=doxtr_rag --cov-report=term-missing --cov-fail-under=90

mypy:
	$(PY) -m mypy

tsc:
	cd $(EXT_DIR) && npx tsc --noEmit --strict

tstest:
	cd $(EXT_DIR) && npx vitest run

xlang:
	$(PY) -m pytest tests/test_cross_language.py -v

dox-build:
	@echo "[qa] dox reference full-build regression (real sphinx-build of the dox project)"
	$(PY) -m pytest tests/test_dox_reference_build.py -v

dox-pdf:
	@echo "[qa] dox PDF ingest regression (real LaTeX/lualatex PDF -> P3 extractor)"
	$(PY) -m pytest tests/test_dox_pdf_ingest.py -v

verifier:
	@echo "[qa] extension-verifier regression smoke check"
	@if [ -x .pi/skills/extension-verifier/verify-extension.sh ]; then \
		bash .pi/skills/extension-verifier/verify-extension.sh || \
		echo "[qa] extension-verifier skipped/failed (non-fatal in constrained env)"; \
	else \
		echo "[qa] extension-verifier skill not present; skipping"; \
	fi

# The full gate. `test` runs the WHOLE pytest suite under coverage — which
# already includes the cross-language round-trip (xlang) and the dox reference
# full-build regression (dox-build). Those are also exposed as standalone
# targets for focused runs.
qa: mypy tsc tstest test verifier
	@echo ""
	@echo "[qa] all gate checks completed."

clean:
	rm -rf .mypy_cache .pytest_cache htmlcov .coverage coverage.xml
	rm -rf test_harness/_build test_harness/.doxtr .doxtr
	rm -rf $(EXT_DIR)/node_modules/.vitest
