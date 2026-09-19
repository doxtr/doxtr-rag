# doxtr-rag Test Harness — Makefile
.PHONY: help test clean build

help:
	@echo "doxtr-rag Test Harness"
	@echo "  make test    Build the harness Sphinx project + smoke assertions"
	@echo "  make build   Build only (HTML)"
	@echo "  make clean   Remove _build/ and scratch .doxtr/"

test:
	python test_runner.py

build:
	python test_runner.py

clean:
	python test_runner.py --clean || true
	rm -rf _build .doxtr
