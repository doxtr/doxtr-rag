"""Sphinx event lifecycle dispatch for doxtr-rag.

This module is the single place that connects Sphinx's build events to the
doxtr-rag pipeline. It is intentionally defensive: any component that is not
yet available (during incremental development) or that fails at runtime
(CONVENTIONS §Graceful Degradation) must never crash ``sphinx-build``.

The pipeline stores per-build state on the ``app`` object (NOT ``app.env``,
which Sphinx pickles to ``environment.pickle`` and merges across parallel-read
workers) under the ``_doxtr_rag_state`` attribute so it survives across the
three events without being serialized. A live pipeline holds non-picklable
resources (SQLite connection, httpx client); keeping it off ``app.env`` avoids
a pickling crash during the read phase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sphinx.application import Sphinx

_STATE_ATTR = "_doxtr_rag_state"


def _get_state(app: "Sphinx") -> dict[str, Any]:
    """Return (creating if needed) the per-build pipeline state dict.

    State lives on ``app`` (never ``app.env``) so the pipeline object — which
    holds non-picklable resources once P5 attaches it — is never serialized
    into the pickled build environment.
    """
    state = getattr(app, _STATE_ATTR, None)
    if state is None:
        state = {}
        setattr(app, _STATE_ATTR, state)
    return state


def on_builder_inited(app: "Sphinx") -> None:
    """Load config, resolve mounts, initialize stores, scan external mounts.

    Wave A ships the safe no-op scaffold; P5 (storage/ingestion) owns and
    implements ``doxtr_rag.pipeline.Pipeline`` — the internal (non-contract)
    seam this glue drives. P5 MUST provide a ``Pipeline`` class with:

        Pipeline.from_app(app) -> Pipeline
        pipeline.builder_inited() -> None
        pipeline.doctree_resolved(doctree, docname) -> None
        pipeline.build_finished(exception) -> None

    P5 may either implement that module or reshape this wiring (one writer per
    file). Until then the ImportError branch is a safe no-op.
    """
    state = _get_state(app)
    try:
        from doxtr_rag.pipeline import Pipeline

        pipeline = Pipeline.from_app(app)
        pipeline.builder_inited()
        state["pipeline"] = pipeline
    except ImportError:
        # Pipeline not yet implemented (pre-P5). Safe no-op.
        logger.debug("doxtr-rag pipeline not available; skipping builder-inited")
    except Exception as exc:  # noqa: BLE001 - fail-safe, never crash the build
        logger.warning("doxtr-rag builder-inited degraded: %s", exc)


def on_doctree_resolved(app: "Sphinx", doctree: Any, docname: str) -> None:
    """Harvest, classify, cache-check and buffer chunks for one document."""
    state = _get_state(app)
    pipeline = state.get("pipeline")
    if pipeline is None:
        return
    try:
        pipeline.doctree_resolved(doctree, docname)
    except Exception as exc:  # noqa: BLE001 - fail-safe
        logger.warning("doxtr-rag doctree-resolved degraded for %s: %s", docname, exc)


def on_build_finished(app: "Sphinx", exception: Exception | None) -> None:
    """Upsert, prune, flush the audit log and close resources."""
    state = _get_state(app)
    pipeline = state.get("pipeline")
    if pipeline is None:
        return
    try:
        pipeline.build_finished(exception)
    except Exception as exc:  # noqa: BLE001 - fail-safe
        logger.warning("doxtr-rag build-finished degraded: %s", exc)
    finally:
        state.pop("pipeline", None)
