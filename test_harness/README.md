# doxtr-rag Test Harness

A real Sphinx project used for integration and regression testing of the
`doxtr-rag` extension. It mirrors the structure of the
`doxtr-pdf-theme-core` reference harness.

## Layout

- `conf.py` — base Sphinx config with the doxtr-rag extension enabled and the
  stores pointed at an embedded scratch `.doxtr/`.
- `source/` — the documentation tree:
  - `index.rst`, `_test_cases/` — `SHARED_ORG` candidate docs.
  - `source/` — an explicitly-declared `SHARED_ORG` mount.
  - `noter/`, `NDA/` — mounts that always route to `LOCAL_PRIVATE`.
  - `_static/` — static assets.
- `conf_overrides/` — per-scenario config overrides (e.g. HTTP store mode,
  read-only shared store).
- `assertions.py` — reusable assertion helpers (scope isolation, audit scrub).
- `test_runner.py` — CI entry point; builds and smoke-asserts.

## Running

```bash
cd test_harness
python test_runner.py          # build + assertions
python test_runner.py --clean  # clean scratch state first
```

The full cross-language round-trip (Python ingest → TS query) lives in the P7
integration test, not here.
