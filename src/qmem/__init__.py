"""
qmem
----

Lightweight client for embedding, ingesting, and retrieving records from Qdrant.

Public API:
- QMem: main client
- IngestItem, RetrievalResult: data models
- QMemConfig: configuration model
- create / ingest / retrieve: module-level helpers with sensible defaults
"""

from __future__ import annotations

from importlib import metadata as _metadata
from typing import Iterable, Optional, Union

# Core low-level client & models
from .client import QMem
from .config import QMemConfig
from .schemas import IngestItem, RetrievalResult

# Low-level API functions (used by module-level helpers)
from .api import (
    create as _api_create,
    ingest as _api_ingest,
    ingest_from_file as _api_ingest_from_file,
    retrieve as _api_retrieve,
)

__all__ = (
    "QMem",
    "IngestItem",
    "RetrievalResult",
    "QMemConfig",
    "create",
    "ingest",
    "retrieve",
    "format_results_table",
    "__version__",
)

# -----------------------------
# Version
# -----------------------------
try:
    # Populated when installed as a package (pip/poetry)
    __version__: str = _metadata.version("qmem")
except _metadata.PackageNotFoundError:
    # Fallback for editable/local use
    __version__ = "0.0.6"

# -----------------------------
# Defaults & state
# -----------------------------
_DEFAULT_COLLECTION: Optional[str] = None
_DEFAULT_DIM: int = 1536            # default vector dim if not specified
_DEFAULT_DISTANCE: str = "cosine"   # default distance if not specified


def _ensure_collection_set(name: Optional[str]) -> str:
    """Resolve the active collection for module-level helpers."""
    global _DEFAULT_COLLECTION
    if name:
        _DEFAULT_COLLECTION = name
        return name
    if not _DEFAULT_COLLECTION:
        raise ValueError("No collection selected. Call create(collection_name=...) first.")
    return _DEFAULT_COLLECTION


# -----------------------------
# Pretty table formatter
# -----------------------------
def format_results_table(
    results,
    *,
    max_query_len: int = 75,
    max_resp_len: int = 60,
) -> str:
    """Build a Unicode table with columns: # | score | query | response."""
    def trunc(s: Optional[str], n: int) -> str:
        s = "" if s is None else str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    header = ["#", "score", "query", "response"]
    w0 = max(2, len(header[0]))
    w1 = max(5, len(header[1]))
    w2 = max(75, len(header[2]), max_query_len)
    w3 = max(60, len(header[3]), max_resp_len)

    top = f"┌{'─'*w0}┬{'─'*w1}┬{'─'*w2}┬{'─'*w3}┐"
    mid = f"├{'─'*w0}┼{'─'*w1}┼{'─'*w2}┼{'─'*w3}┤"
    bot = f"└{'─'*w0}┴{'─'*w1}┴{'─'*w2}┴{'─'*w3}┘"

    def row(a, b, c, d) -> str:
        return (
            f"│{str(a).rjust(w0)}"
            f"│{str(b).rjust(w1)}"
            f"│{str(c).ljust(w2)}"
            f"│{str(d).ljust(w3)}│"
        )

    lines = [top, row(*header), mid]
    for i, r in enumerate(results, 1):
        score = f"{getattr(r, 'score', 0):.4f}"
        payload = getattr(r, "payload", {}) or {}
        q = trunc(payload.get("query"), max_query_len)       # strictly 'query'
        resp = trunc(payload.get("response"), max_resp_len)  # strictly 'response'
        lines.append(row(i, score, q, resp))
    lines.append(bot)
    return "\n".join(lines)


# -----------------------------
# Public module-level helpers
# -----------------------------
def create(
    *,
    collection_name: str,
    dim: Optional[int] = None,
    distance_metric: Union[str, object] = None,
    cfg: Optional[QMemConfig] = None,
) -> None:
    """
    qm.create(collection_name="test_learn", dim=1536, distance_metric="cosine")

    Defaults applied if not mentioned:
      - dim = 1536
      - distance_metric = "cosine"

    Also sets the default collection for subsequent qm.ingest / qm.retrieve calls.
    """
    global _DEFAULT_COLLECTION
    if dim is None:
        dim = _DEFAULT_DIM
    if distance_metric is None:
        distance_metric = _DEFAULT_DISTANCE
    _api_create(collection_name, cfg=cfg, dim=dim, distance=distance_metric)
    _DEFAULT_COLLECTION = collection_name


def ingest(
    *,
    file: Optional[str] = None,
    records: Optional[Iterable[dict]] = None,
    embed_field: Optional[str] = None,
    payload_field: Optional[str] = None,
    include_embed_in_payload: bool = True,   # include embedded text in payload by default
    collection_name: Optional[str] = None,
    cfg: Optional[QMemConfig] = None,
) -> int:
    """
    qm.ingest(file="/path/data.jsonl", embed_field="sql_query", payload_field="query,response")

    Rules:
      - embed_field is REQUIRED
      - Provide either file= path OR records=[...]
      - If payload_field is omitted -> keep ALL payload except the embedded field (default)
      - include_embed_in_payload=True keeps the embedded field text in payload
    """
    if not embed_field:
        raise ValueError("embed_field is required")

    coll = _ensure_collection_set(collection_name)
    payload_keys = [s.strip() for s in (payload_field or "").split(",") if s.strip()] or None

    if file:
        return _api_ingest_from_file(
            coll,
            file,
            embed_field=embed_field,
            cfg=cfg,
            payload_keys=payload_keys,
            include_embed_in_payload=include_embed_in_payload,
        )

    if records is not None:
        return _api_ingest(
            coll,
            records,
            embed_field=embed_field,
            cfg=cfg,
            payload_keys=payload_keys,
            include_embed_in_payload=include_embed_in_payload,
        )

    raise ValueError("Provide either file= path or records=[...]")

def retrieve(
    *,
    query: str,
    top_k: int = 5,
    collection_name: Optional[str] = None,
    as_table: bool = True,
    cfg: Optional[QMemConfig] = None,
):
    """
    qm.retrieve(query="...", top_k=5)

    Returns:
      - pretty table string by default (as_table=True)
      - raw results list if as_table=False
    """
    if not query:
        raise ValueError("query is required")

    coll = _ensure_collection_set(collection_name)
    results = _api_retrieve(coll, query, k=top_k, cfg=cfg)
    return format_results_table(results) if as_table else results
