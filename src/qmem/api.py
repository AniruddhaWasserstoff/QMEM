from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

from qdrant_client import models as qmodels

from .client import QMem
from .config import CONFIG_PATH, QMemConfig
from .schemas import IngestItem, RetrievalResult

__all__ = [
    "create",
    "ingest",
    "ingest_from_file",
    "retrieve",
    "retrieve_by_filter",
    "mongo",  # NEW: mirror existing Qdrant collection -> MongoDB
]

# -----------------------------
# Internals
# -----------------------------

_DISTANCE: Dict[str, qmodels.Distance] = {
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

    Flat payload behavior:
      - Keep ALL original keys flat at top level (no 'extra' wrapper).
      - Ensure the chosen `embed_field` is present at top level when available.
      - Convenience fields (query/response/sql_query/doc_id/graph/tags) pass through if present.
    """
    items: List[IngestItem] = []
    for d in records:
        # Start with pass-through of the whole record (flat)
        known = dict(d)

        # Control field for which key to embed
        known["embed_field"] = embed_field

        # Ensure the embed text is available at top-level for the client
        if embed_field and embed_field in d:
            known[embed_field] = d[embed_field]

        # Convenience keys (may already exist; that's fine)
        for k in ("query", "response", "sql_query", "doc_id", "graph", "tags"):
            if k in d:
                known[k] = d[k]

        # IMPORTANT: Do NOT create/keep 'extra'
        known.pop("extra", None)

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

    Behavior:
      - If `dim` is provided, it overrides `cfg.embed_dim`, is persisted to config,
        and both the collection vector size and embedder dimension will use this value.
      - If `dim` is omitted, falls back to `cfg.embed_dim` (or 1536 if unset).

    Args:
        collection: Collection name.
        cfg: Optional QMemConfig; loaded from CONFIG_PATH if not provided.
        dim: Vector size; overrides and persists cfg.embed_dim when provided.
        distance: "cosine" | "dot" | "euclid" (or qmodels.Distance).
    """
    cfg = cfg or QMemConfig.load(CONFIG_PATH)

    # Resolve vector dimension and persist override if provided
    if dim is None:
        vec_dim = cfg.embed_dim or 1536
    else:
        vec_dim = int(dim)
        if cfg.embed_dim != vec_dim:
            cfg.embed_dim = vec_dim
            cfg.save(CONFIG_PATH)  # persist so the embedder and future ops match

    # Normalize distance
    if isinstance(distance, str):
        key = distance.strip().lower()
        if key not in _DISTANCE:
            raise ValueError(f"Invalid distance: {distance!r}. Choose from: {', '.join(_DISTANCE)}")
        dist = _DISTANCE[key]
    else:
        dist = distance

    # Build client AFTER cfg.embed_dim may have been updated, so embedder matches
    q = QMem(cfg, collection=collection)

    # If the collection already exists, nothing to do
    try:
        q.client.get_collection(collection)
        return
    except Exception:
        pass

    # Create with the resolved dimension & distance
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
        payload_keys: Fields to store in payload; None = keep ALL fields (flat).
        include_embed_in_payload: Also keep the embedded field in payload (if True).

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
        payload_keys: Fields to store in payload; None = keep ALL fields (flat).
        include_embed_in_payload: Also keep the embedded field in payload (if True).

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
) -> List[RetrievalResult]:
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


def retrieve_by_filter(
    collection: str,
    *,
    filter: Union[dict, qmodels.Filter],
    k: int = 100,
    query: Optional[str] = None,
    cfg: Optional[QMemConfig] = None,
) -> List[RetrievalResult]:
    """
    Hybrid (query + filter) when `query` is provided; otherwise payload-only scroll.

    Returns:
        list[RetrievalResult] (first page up to k items).
    """
    cfg = cfg or QMemConfig.load(CONFIG_PATH)
    q = QMem(cfg, collection=collection)
    q.client.get_collection(collection)  # ensure exists

    if query:
        return q.search_filtered(query, top_k=k, query_filter=filter)

    results, _ = q.scroll_filter(query_filter=filter, limit=k)
    return results


# -----------------------------
# NEW: programmatic Mongo mirror
# -----------------------------

def mongo(
    *,
    collection_name: str,
    fields: Optional[Sequence[str]] = None,          # None/[] => FULL payload
    mongo_uri: str = "mongodb://127.0.0.1:27017",
    mongo_db: str = "qmem",
    mongo_collection: Optional[str] = None,          # defaults to collection_name
    batch_size: int = 1000,
    max_docs: Optional[int] = None,
    cfg: Optional[QMemConfig] = None,
) -> int:
    """
    Mirror an existing Qdrant collection's payloads into MongoDB.

    Args:
        collection_name: Qdrant collection to read from (must already exist).
        fields: Subset of payload keys to store in Mongo.
                None or empty => mirror FULL payload.
        mongo_uri: Mongo connection string.
        mongo_db: Target Mongo database.
        mongo_collection: Target collection (defaults to collection_name).
        batch_size: Qdrant scroll page size (performance/memory trade-off).
        max_docs: Optional total cap on mirrored documents (None = all).
        cfg: Optional QMemConfig; loaded from CONFIG_PATH if not provided.

    Returns:
        Number of documents mirrored to Mongo.
    """
    cfg = cfg or QMemConfig.load(CONFIG_PATH)
    q = QMem(cfg, collection=collection_name)

    # Ensure the source collection exists
    try:
        q.client.get_collection(collection_name)
    except Exception as e:
        raise RuntimeError(f"No such Qdrant collection: {collection_name}") from e

    mongo_keys: Optional[Set[str]] = set(fields) if fields else None
    coll = mongo_collection or collection_name

    return q.mirror_to_mongo(
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        mongo_coll=coll,
        mongo_keys=mongo_keys,
        batch_size=batch_size,
        max_docs=max_docs,
    )
