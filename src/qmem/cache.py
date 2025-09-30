# qmem/cache.py
from __future__ import annotations
import json, time, hashlib, threading
from typing import Any, Dict, Optional, Tuple, List

from .config import QMemConfig
from .client import QMem

def _canonical_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))
    except Exception:
        return json.dumps(str(obj))

def make_exact_key(*, collection: str, query: str, top_k: int, filt: Optional[dict]) -> Tuple[str, str]:
    fhash = hashlib.sha1(_canonical_json(filt).encode("utf-8")).hexdigest() if filt else "-"
    base = json.dumps({"c": collection, "q": query, "k": top_k, "f": fhash}, sort_keys=True, separators=(",", ":"))
    ek = hashlib.sha1(base.encode("utf-8")).hexdigest()
    return ek, fhash

class LocalCache:
    """In-memory TTL cache (no DB work)."""
    def __init__(self, ttl_sec: int = 3600, max_items: int = 5000):
        self.ttl = ttl_sec
        self.max = max_items
        self._lock = threading.Lock()
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            it = self._store.get(key)
            if not it:
                return None
            ts, val = it
            if now - ts > self.ttl:
                self._store.pop(key, None)
                return None
            return val

    def set(self, key: str, val: Any) -> None:
        now = time.time()
        with self._lock:
            if len(self._store) >= self.max:
                # drop oldest
                oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                self._store.pop(oldest, None)
            self._store[key] = (now, val)

class SemanticIndex:
    """
    Stores (query_text, cached answer payload) in a sibling collection: <base>__semcache.
    We search this collection semantically to reuse answers for paraphrased queries.
    """
    def __init__(self, cfg: QMemConfig, base_collection: str):
        self.cfg = cfg
        self.name = f"{base_collection}__semcache"
        self.q = QMem(cfg, collection=self.name)
        # Ensure cache collection exists (uses embedder dim from cfg / backend)
        self.q.ensure_collection(create_if_missing=True)

        # Thresholds: Qdrant scores are similarity (higher better); Chroma returns distances (lower better).
        self._backend = getattr(self.q, "_backend", "qdrant")
        self._sim_close = 0.97            # for cosine similarity (Qdrant)
        self._dist_close = 0.08           # for distance (Chroma)

    def _is_close(self, score: float) -> bool:
        if self._backend == "chroma":
            return score <= self._dist_close
        return score >= self._sim_close

    def lookup(self, query: str, *, top_k: int, filter_hash: str) -> Optional[dict]:
        # Search by the new query; top match must have same top_k & filter_hash
        hits = self.q.search(query, top_k=3)
        for h in hits:
            pl = h.payload or {}
            if pl.get("top_k") == top_k and pl.get("filter_hash") == filter_hash and self._is_close(h.score):
                return pl.get("answer")  # stored as a dict
        return None

    def store(self, *, query: str, top_k: int, filter_hash: str, answer: dict) -> None:
        from .schemas import IngestItem
        # We embed 'query_text' so the index learns paraphrases; we keep the answer in payload.
        item = IngestItem(embed_field="query_text",
                          query_text=query,
                          top_k=top_k,
                          filter_hash=filter_hash,
                          answer=answer)
        self.q.ingest([item], include_embed_in_payload=False)
