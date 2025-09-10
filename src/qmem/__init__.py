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
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

# Core low-level client & models
from .client import QMem
from .config import QMemConfig, FILTERS_DIR
from .schemas import IngestItem, RetrievalResult

# Low-level API functions (used by module-level helpers)
from .api import (
    create as _api_create,
    ingest as _api_ingest,
    ingest_from_file as _api_ingest_from_file,
    retrieve as _api_retrieve,
    retrieve_by_filter as _api_retrieve_by_filter,
    mongo as _api_mongo,  # NEW
)

__all__ = (
    "QMem",
    "IngestItem",
    "RetrievalResult",
    "QMemConfig",
    "create",
    "ingest",
    "retrieve",
    "filter",
    "retrieve_filter",
    "mongo",              # NEW
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
    __version__ = "0.1.3"

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


def _resolve_filter_arg(filter_arg: Union[dict, str, Path]) -> dict:
    """
    Accepts:
      - dict (returned as-is)
      - str/Path:
          * absolute/relative JSON path, or
          * short name 'myfilter' resolving to ./.qmem/filters/myfilter.json
    """
    if isinstance(filter_arg, dict):
        return filter_arg

    if isinstance(filter_arg, (str, Path)):
        p = Path(filter_arg)
        # If it's a bare name (no suffix and not an explicit path), map to FILTERS_DIR/name.json
        if not p.suffix and not str(p).startswith(("/", "./", "../")):
            p = FILTERS_DIR / f"{p}.json"
        if not p.exists():
            raise FileNotFoundError(f"Filter file not found: {p}")
        text = p.read_text(encoding="utf-8")
        import json as _json
        obj = _json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(f"Filter file must contain a JSON object: {p}")
        return obj

    raise TypeError("filter_json must be a dict, a path string, or a Path")


# -----------------------------
# Pretty table formatter
# -----------------------------
def format_results_table(
    results: Sequence[RetrievalResult],
    *,
    max_query_len: int = 75,   # legacy knobs kept; applied to the first payload column
    max_resp_len: int = 60,    # used for subsequent payload columns
    show_score: bool = True,
    show_keys: Optional[Sequence[str]] = None,  # None -> infer ALL payload keys
) -> str:
    """
    Build a Unicode table.

    Defaults:
      - show_score=True → includes a 'score' column
      - show_keys=None  → infer ALL scalar payload keys from results (frequency desc, then name)

    Example:
      format_results_table(results, show_keys=['title', 'type'])
    """
    def trunc(s: Optional[str], n: int) -> str:
        s = "" if s is None else str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    # Infer keys if not provided: union of scalar payload keys across results,
    # ordered by (frequency desc, name asc).
    if show_keys is None:
        from collections import Counter
        freq = Counter()
        for r in results or []:
            payload = getattr(r, "payload", {}) or {}
            for k, v in payload.items():
                if isinstance(v, (str, int, float, bool)) and (str(v).strip() if isinstance(v, str) else True):
                    freq[k] += 1
        keys: List[str] = [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    else:
        keys = list(show_keys)

    # Decide column widths:
    # - "#"   -> 2
    # - "score" -> 5
    # - payload cols: first gets max(20, max_query_len) to keep a wide lead column; others use max(20, max_resp_len)
    first_payload_w = max(20, max_query_len) if keys else 0
    payload_widths = [(first_payload_w if i == 0 else max(20, max_resp_len)) for i in range(len(keys))]

    if show_score:
        widths = [2, 5] + payload_widths
        header = ["#", "score"] + keys
    else:
        widths = [2] + payload_widths
        header = ["#"] + keys

    # Build borders dynamically
    top = "┌" + "┬".join("─" * w for w in widths) + "┐"
    mid = "├" + "┼".join("─" * w for w in widths) + "┤"
    bot = "└" + "┴".join("─" * w for w in widths) + "┘"

    def row(cells: Sequence[object]) -> str:
        out: List[str] = []
        for i, (cell, w) in enumerate(zip(cells, widths)):
            text = str(cell)
            # Right-justify first two columns (#, score) when score is present;
            # when show_score=False, only '#' is right-justified.
            if (show_score and i <= 1) or (not show_score and i == 0):
                out.append(text.rjust(w))
            else:
                out.append(text.ljust(w))
        return "│" + "│".join(out) + "│"

    # Assemble table
    lines = [top, row(header), mid]
    for idx, r in enumerate(results, 1):
        payload = getattr(r, "payload", {}) or {}
        payload_cells: List[str] = []
        for j, k in enumerate(keys):
            limit = first_payload_w if j == 0 else max(20, max_resp_len)
            payload_cells.append(trunc(payload.get(k), limit))

        if show_score:
            cells: List[object] = [idx, f"{getattr(r, 'score', 0):.4f}"] + payload_cells
        else:
            cells = [idx] + payload_cells

        lines.append(row(cells))

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
    show: Optional[Sequence[str]] = None,  # choose payload columns in the table; None -> ALL
) -> Union[str, List[RetrievalResult]]:
    """
    qm.retrieve(query="...", top_k=5, show=["title","type"])

    Returns:
      - pretty table string by default (as_table=True)
      - raw results list if as_table=False
    """
    if not query:
        raise ValueError("query is required")

    coll = _ensure_collection_set(collection_name)
    results = _api_retrieve(coll, query, k=top_k, cfg=cfg)
    return format_results_table(results, show_keys=show) if as_table else results


def filter(
    *,
    filter_json: Union[dict, str, Path],
    limit: int = 100,
    collection_name: Optional[str] = None,
    as_table: bool = True,
    cfg: Optional[QMemConfig] = None,
    show: Optional[Sequence[str]] = None,   # NEW: choose payload columns; None -> ALL
) -> Union[str, List[RetrievalResult]]:
    """
    qm.filter(filter_json={...} | 'latest' | 'path/to/file.json', limit=100, show=['title','type'])
    Payload-only results by default; prints a table WITHOUT the score column.
    """
    coll = _ensure_collection_set(collection_name)
    filt = _resolve_filter_arg(filter_json)
    results = _api_retrieve_by_filter(coll, filter=filt, k=limit, query=None, cfg=cfg)
    return format_results_table(results, show_score=False, show_keys=show) if as_table else results


def retrieve_filter(
    *,
    query: str,
    filter_json: Union[dict, str, Path],
    top_k: int = 5,
    collection_name: Optional[str] = None,
    as_table: bool = True,
    cfg: Optional[QMemConfig] = None,
    show: Optional[Sequence[str]] = None,   # NEW: choose payload columns; None -> ALL
) -> Union[str, List[RetrievalResult]]:
    """
    qm.retrieve_filter(query, filter_json={...} | 'latest' | 'path/to/file.json', top_k=5, show=['title','type'])
    Hybrid search; prints a table WITHOUT the score column by default.
    """
    if not query:
        raise ValueError("query is required")
    coll = _ensure_collection_set(collection_name)
    filt = _resolve_filter_arg(filter_json)
    results = _api_retrieve_by_filter(coll, filter=filt, k=top_k, query=query, cfg=cfg)
    return format_results_table(results, show_score=False, show_keys=show) if as_table else results


# -----------------------------
# NEW: programmatic Mongo mirror helper
# -----------------------------
def mongo(
    *,
    collection_name: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,          # None/[] => FULL payload
    mongo_uri: str = "mongodb://127.0.0.1:27017",
    mongo_db: str = "qmem",
    mongo_collection: Optional[str] = None,          # defaults to collection_name
    batch_size: int = 1000,
    max_docs: Optional[int] = None,
    cfg: Optional[QMemConfig] = None,
) -> int:
    """
    qm.mongo(collection_name="...", fields=["title","type"], mongo_db="qmem_payload_db", mongo_collection="qmem_payload")

    Mirrors payloads from an existing Qdrant collection into MongoDB.
    If collection_name is omitted, uses the last collection set via qm.create(...).
    """
    coll = _ensure_collection_set(collection_name)
    return _api_mongo(
        collection_name=coll,
        fields=fields,
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        mongo_collection=mongo_collection,
        batch_size=batch_size,
        max_docs=max_docs,
        cfg=cfg,
    )
