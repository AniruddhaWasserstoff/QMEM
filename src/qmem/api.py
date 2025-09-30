from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

from qdrant_client import models as qmodels

from .client import QMem
from .config import CONFIG_PATH, QMemConfig
from .schemas import IngestItem, RetrievalResult

# NEW: caching imports
from .cache import LocalCache, SemanticIndex, make_exact_key

__all__ = [
    "create",
    "ingest",
    "ingest_from_file",
    "retrieve",
    "retrieve_by_filter",
    "mongo",  # mirror existing collection -> MongoDB (Qdrant or Chroma backend)
]

# -----------------------------
# Internals
# -----------------------------

_DISTANCE: Dict[str, qmodels.Distance] = {
    "cosine": qmodels.Distance.COSINE,
    "dot": qmodels.Distance.DOT,
    "euclid": qmodels.Distance.EUCLID,
}

# NEW: module-level exact-key cache (in-memory TTL; overridable via env)
_LOCAL_CACHE = LocalCache(
    ttl_sec=int(os.getenv("QMEM_CACHE_TTL", "1800")),   # default 30 minutes
    max_items=int(os.getenv("QMEM_CACHE_MAX", "5000")),
)


def _normalize_payload_keys(keys: Optional[Sequence[str]]) -> Optional[Set[str]]:
    """Normalize a sequence of payload keys to a unique, trimmed set (or None)."""
    if keys is None:
        return None
    return {k.strip() for k in keys if k and k.strip()}


def _items(records: Iterable[dict], embed_field: Optional[str]) -> List[IngestItem]:
    """
    Convert raw dict records into IngestItem objects.
    """
    items: List[IngestItem] = []
    for d in records:
        known = dict(d)
        known["embed_field"] = embed_field
        if embed_field and embed_field in d:
            known[embed_field] = d[embed_field]
        for k in ("query", "response", "sql_query", "doc_id", "graph", "tags"):
            if k in d:
                known[k] = d[k]
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


# Helper: normalize qdrant filter to dict safely (pydantic v1/v2 compatibility)
def _filter_to_dict(f: Union[dict, qmodels.Filter]) -> dict:
    if isinstance(f, dict):
        return f
    # qdrant pydantic v1 uses .dict(), v2 uses .model_dump()
    if hasattr(f, "model_dump"):
        return f.model_dump()
    if hasattr(f, "dict"):
        return f.dict()
    # last resort
    return json.loads(json.dumps(f, default=str))


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
    cfg = cfg or QMemConfig.load(CONFIG_PATH)
    if dim is None:
        vec_dim = cfg.embed_dim or 1536
    else:
        vec_dim = int(dim)
        if cfg.embed_dim != vec_dim:
            cfg.embed_dim = vec_dim
            cfg.save(CONFIG_PATH)
    q = QMem(cfg, collection=collection)
    backend = getattr(q, "_backend", "qdrant")
    try:
        q.ensure_collection(create_if_missing=False)
        return
    except Exception:
        pass
    if backend == "qdrant":
        if isinstance(distance, str):
            key = distance.strip().lower()
            if key not in _DISTANCE:
                raise ValueError(f"Invalid distance: {distance!r}. Choose from: {', '.join(_DISTANCE)}")
            dist = _DISTANCE[key]
        else:
            dist = distance
        q.ensure_collection(create_if_missing=True, distance=dist, vector_size=vec_dim)
    else:
        q.ensure_collection(create_if_missing=True)


def ingest(
    collection: str,
    records: Iterable[dict],
    *,
    embed_field: Optional[str],
    cfg: Optional[QMemConfig] = None,
    payload_keys: Optional[Sequence[str]] = None,
    include_embed_in_payload: bool = True,
) -> int:
    if not embed_field:
        raise ValueError("embed_field is required")
    cfg = cfg or QMemConfig.load(CONFIG_PATH)
    q = QMem(cfg, collection=collection)
    try:
        q.ensure_collection(create_if_missing=False)
    except Exception as e:
        raise RuntimeError(f"No such collection: {collection}") from e
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
    Two-layer cached retrieve:
    1) Exact-key cache (same query/top_k) → instant hit, no DB
    2) Semantic cache (<collection>__semcache) → reuse prior answer if close enough
    3) Fallback to live vector search, then populate both caches
    """
    if not query:
        raise ValueError("query is required")

    cfg = cfg or QMemConfig.load(CONFIG_PATH)

    # ---- 1) Exact-key cache
    ek, fhash = make_exact_key(collection=collection, query=query, top_k=k, filt=None)
    hit = _LOCAL_CACHE.get(ek)
    if hit:
        return [RetrievalResult(**r) for r in hit["results"]]

    # ---- 2) Semantic cache
    sem = SemanticIndex(cfg, base_collection=collection)
    sem_hit = sem.lookup(query, top_k=k, filter_hash=fhash)
    if sem_hit:
        return [RetrievalResult(**r) for r in sem_hit["results"]]

    # ---- 3) Live vector search
    q = QMem(cfg, collection=collection)
    try:
        q.ensure_collection(create_if_missing=False)
    except Exception as e:
        raise RuntimeError(f"No such collection: {collection}") from e

    results = q.search(query, top_k=k)

    # Persist to caches
    payload = {"results": [r.model_dump() for r in results], "version": 1, "ts": time.time()}
    _LOCAL_CACHE.set(ek, payload)
    sem.store(query=query, top_k=k, filter_hash=fhash, answer=payload)

    return results


def retrieve_by_filter(
    collection: str,
    *,
    filter: Union[dict, qmodels.Filter],
    k: int = 100,
    query: Optional[str] = None,
    cfg: Optional[QMemConfig] = None,
) -> List[RetrievalResult]:
    """
    Cached filtered retrieve:
    - If query is None: do a filtered scroll (no semantic layer, no exact cache)
    - Else: apply the same two-layer cache strategy keyed by (collection, query, k, filter)
    """
    cfg = cfg or QMemConfig.load(CONFIG_PATH)
    q = QMem(cfg, collection=collection)
    try:
        q.ensure_collection(create_if_missing=False)
    except Exception as e:
        raise RuntimeError(f"No such collection: {collection}") from e

    # No query → just filter scan; skip caching entirely
    if not query:
        results, _ = q.scroll_filter(query_filter=filter, limit=k)
        return results

    # Normalize filter to dict for hashing
    filt_dict = _filter_to_dict(filter)

    # ---- 1) Exact-key cache
    ek, fhash = make_exact_key(collection=collection, query=query, top_k=k, filt=filt_dict)
    hit = _LOCAL_CACHE.get(ek)
    if hit:
        return [RetrievalResult(**r) for r in hit["results"]]

    # ---- 2) Semantic cache
    sem = SemanticIndex(cfg, base_collection=collection)
    sem_hit = sem.lookup(query, top_k=k, filter_hash=fhash)
    if sem_hit:
        return [RetrievalResult(**r) for r in sem_hit["results"]]

    # ---- 3) Live filtered search
    results = q.search_filtered(query, top_k=k, query_filter=filter)

    # Persist to caches
    payload = {"results": [r.model_dump() for r in results], "version": 1, "ts": time.time()}
    _LOCAL_CACHE.set(ek, payload)
    sem.store(query=query, top_k=k, filter_hash=fhash, answer=payload)

    return results


# -----------------------------
# Programmatic Mongo mirror
# -----------------------------

def mongo(
    *,
    collection_name: str,
    fields: Optional[Sequence[str]] = None,
    mongo_uri: str = "mongodb://127.0.0.1:27017",
    mongo_db: str = "qmem",
    mongo_collection: Optional[str] = None,
    batch_size: int = 1000,
    max_docs: Optional[int] = None,
    cfg: Optional[QMemConfig] = None,
) -> int:
    """
    Mirror an existing collection's payloads into MongoDB (supports Qdrant and Chroma backends).
    """
    cfg = cfg or QMemConfig.load(CONFIG_PATH)
    q = QMem(cfg, collection=collection_name)
    try:
        q.ensure_collection(create_if_missing=False)
    except Exception as e:
        raise RuntimeError(f"No such collection: {collection_name}") from e
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