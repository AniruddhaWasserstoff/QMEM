from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Union

from qdrant_client import models as qmodels

from .client import QMem
from .config import CONFIG_PATH, QMemConfig
from .schemas import IngestItem


# -----------------------------
# Internals
# -----------------------------

_DISTANCE = {
    "cosine": qmodels.Distance.COSINE,
    "dot": qmodels.Distance.DOT,
    "euclid": qmodels.Distance.EUCLID,
}


def _normalize_payload_keys(keys: Optional[Sequence[str]]) -> Optional[Set[str]]:
    """Normalize a sequence of payload keys to a unique, trimmed set (or None)."""
    if keys is None:
        return None
    return {k.strip() for k in keys if k and k.strip()}


def _items(records: Iterable[dict], embed_field: Optional[str]) -> List[IngestItem]:
    """
    Convert raw dict records into IngestItem objects.

    - Keeps the chosen `embed_field` at top-level (so the client can embed it).
    - Copies common text fields when present (query/response/sql_query/doc_id).
    - Remaining keys are placed under `extra`.
    """
    items: List[IngestItem] = []
    for d in records:
        known = {
            "vector": d.get("vector"),
            "embed_field": embed_field,
            "graph": d.get("graph"),
            "tags": d.get("tags"),
        }
        # Ensure the embed text is available at top-level for the client
        if embed_field and embed_field in d:
            known[embed_field] = d[embed_field]

        # Copy well-known text fields if provided
        for k in ("query", "response", "sql_query", "doc_id"):
            if k in d:
                known[k] = d[k]

        # Everything else -> extra
        extra = {k: v for k, v in d.items() if k not in known}
        if extra:
            known["extra"] = extra

        items.append(IngestItem(**known))
    return items


def _read_json_or_jsonl(path: Union[str, Path]) -> List[dict]:
    """Read .jsonl or .json into a list of dicts."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    obj = json.loads(text)
    return obj if isinstance(obj, list) else [obj]


# -----------------------------
# Public API (low-level core)
# -----------------------------

def create(
    collection: str,
    *,
    cfg: Optional[QMemConfig] = None,
    dim: Optional[int] = None,
    distance: Union[str, qmodels.Distance] = "cosine",
) -> None:
    """
    Create a collection in Qdrant if it doesn't already exist.

    Args:
        collection: Collection name.
        cfg: Optional QMemConfig; loaded from CONFIG_PATH if not provided.
        dim: Vector size; falls back to cfg.embed_dim or 1536.
        distance: "cosine" | "dot" | "euclid" (or qmodels.Distance).
    """
    cfg = cfg or QMemConfig.load(CONFIG_PATH)
    q = QMem(cfg, collection=collection)

    try:
        q.client.get_collection(collection)  # exists -> no-op
        return
    except Exception:
        pass

    dist = _DISTANCE[distance] if isinstance(distance, str) else distance
    vec_dim = dim if dim is not None else (cfg.embed_dim or 1536)

    q.ensure_collection(
        create_if_missing=True,
        distance=dist,
        vector_size=vec_dim,
    )


def ingest(
    collection: str,
    records: Iterable[dict],
    *,
    embed_field: Optional[str],
    cfg: Optional[QMemConfig] = None,
    payload_keys: Optional[Sequence[str]] = None,
    include_embed_in_payload: bool = True,
) -> int:
    """
    Ingest in-memory records into a collection.

    Args:
        collection: Collection name.
        records: Iterable of dict rows.
        embed_field: Field whose text to embed (required unless row has 'vector').
        cfg: Optional QMemConfig; loaded if not provided.
        payload_keys: Fields to store in payload; None = keep all except embedded field.
        include_embed_in_payload: Also keep the embedded field in payload.

    Returns:
        Number of upserted items.
    """
    if not embed_field:
        raise ValueError("embed_field is required")

    cfg = cfg or QMemConfig.load(CONFIG_PATH)
    q = QMem(cfg, collection=collection)
    q.client.get_collection(collection)  # ensure exists

    items = _items(records, embed_field)
    return q.ingest(
        items,
        payload_keys=_normalize_payload_keys(payload_keys),
        include_embed_in_payload=include_embed_in_payload,
    )


def ingest_from_file(
    collection: str,
    path: Union[str, Path],
    *,
    embed_field: Optional[str],
    cfg: Optional[QMemConfig] = None,
    payload_keys: Optional[Sequence[str]] = None,
    include_embed_in_payload: bool = True,
) -> int:
    """
    Ingest records from a .jsonl or .json file on disk.

    Args:
        collection: Collection name.
        path: File path (.jsonl or .json).
        embed_field: Field whose text to embed (required).
        cfg: Optional QMemConfig.
        payload_keys: Fields to store in payload; None = keep all except embedded field.
        include_embed_in_payload: Also keep the embedded field in payload.

    Returns:
        Number of upserted items.
    """
    if not embed_field:
        raise ValueError("embed_field is required")

    records = _read_json_or_jsonl(path)
    return ingest(
        collection,
        records,
        embed_field=embed_field,
        cfg=cfg,
        payload_keys=payload_keys,
        include_embed_in_payload=include_embed_in_payload,
    )


def retrieve(
    collection: str,
    query: str,
    *,
    k: int = 5,
    cfg: Optional[QMemConfig] = None,
):
    """
    Vector search for top-k results.

    Args:
        collection: Collection name.
        query: Query text to embed and search.
        k: Number of results (default 5).
        cfg: Optional QMemConfig.

    Returns:
        A list of RetrievalResult objects.
    """
    if not query:
        raise ValueError("query is required")

    cfg = cfg or QMemConfig.load(CONFIG_PATH)
    q = QMem(cfg, collection=collection)
    q.client.get_collection(collection)  # ensure exists
    return q.search(query, top_k=k)
