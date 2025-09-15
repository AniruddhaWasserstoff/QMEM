
from __future__ import annotations

import logging as log
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from uuid import uuid4

from qdrant_client import QdrantClient, models as qmodels

# Chroma is optional; required only when vector_store == "chroma"
try:
    import chromadb  # type: ignore
except Exception:
    chromadb = None  # checked at runtime

from .config import QMemConfig
from .embedder import build_embedder
from .schemas import IngestItem, RetrievalResult

__all__ = ["QMem"]

logger = log.getLogger(__name__)


# =============================================================================
# Qdrant implementation (existing logic, preserved)
# =============================================================================
class _QMemQdrant:
    """Minimal client for ingest & retrieval against a single Qdrant collection."""

    # -----------------------------
    # Init
    # -----------------------------
    def __init__(self, cfg: QMemConfig, collection: Optional[str] = None) -> None:
        """
        Args:
            cfg: Configuration containing API keys and embedding settings.
            collection: Target collection name; defaults to cfg.default_collection.
        """
        self.cfg = cfg
        self.collection = collection or cfg.default_collection
        self.client = QdrantClient(
            url=cfg.qdrant_url,
            api_key=cfg.qdrant_api_key,
            timeout=20.0,
            prefer_grpc=False,
        )
        self.embedder = build_embedder(cfg)

    # ---------------------------------------------------------------------
    # Collection management
    # ---------------------------------------------------------------------
    def ensure_collection(
        self,
        *,
        create_if_missing: bool,
        distance: qmodels.Distance = qmodels.Distance.COSINE,
        vector_size: Optional[int] = None,
    ) -> None:
        """
        Ensure the current collection exists (optionally creating it).
        """
        name = self._require_collection()
        try:
            self.client.get_collection(name)
            return
        except Exception:
            if not create_if_missing:
                raise

        dim = int(vector_size or self.embedder.dim)
        logger.info("Creating collection %s (dim=%s, distance=%s)", name, dim, distance)
        self.client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(size=dim, distance=distance),
            on_disk_payload=True,
            hnsw_config=qmodels.HnswConfigDiff(on_disk=True),
            optimizers_config=qmodels.OptimizersConfigDiff(default_segment_number=2),
            shard_number=1,
        )

    # ---------------------------------------------------------------------
    # Payload index helpers
    # ---------------------------------------------------------------------
    _INDEX_TYPE_MAP: Dict[str, qmodels.PayloadSchemaType] = {
        "keyword": qmodels.PayloadSchemaType.KEYWORD,
        "integer": qmodels.PayloadSchemaType.INTEGER,
        "int": qmodels.PayloadSchemaType.INTEGER,
        "float": qmodels.PayloadSchemaType.FLOAT,
        "double": qmodels.PayloadSchemaType.FLOAT,
        "bool": qmodels.PayloadSchemaType.BOOL,
        "boolean": qmodels.PayloadSchemaType.BOOL,
    }

    def create_payload_index(self, field_name: str, field_type: str) -> None:
        """
        Create a single payload index on `field_name` (keyword|integer|float|bool).
        Safe to call repeatedly; no-ops if already exists.
        """
        name = self._require_collection()
        tkey = (field_type or "").strip().lower()
        schema = self._INDEX_TYPE_MAP.get(tkey)
        if not schema:
            raise ValueError(
                f"Unsupported index type {field_type!r} for field {field_name!r}. "
                f"Use one of: keyword, integer, float, bool"
            )
        try:
            self.client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=schema,
                wait=True,
            )
            logger.info("Created payload index: %s (%s)", field_name, tkey)
        except Exception as e:
            logger.debug("create_payload_index skipped for %s (%s): %s", field_name, tkey, e)

    def ensure_payload_indices(self, schema: Dict[str, str]) -> None:
        """Ensure payload indexes exist for a dict like {'genre':'keyword', 'year':'integer'}."""
        if not schema:
            return
        for field, ftype in schema.items():
            self.create_payload_index(field, ftype)

    # ---------------------------------------------------------------------
    # Ingest
    # ---------------------------------------------------------------------
    def ingest(
        self,
        items: Sequence[IngestItem],
        *,
        batch_size: int = 64,
        payload_keys: Optional[Set[str]] = None,
        include_embed_in_payload: bool = False,
    ) -> int:
        """
        Upsert items into the current collection.
        Embeddings are computed from each item's `embed_field`.
        By default, the embedded text is NOT stored in payload.
        """
        if not items:
            return 0

        name = self._require_collection()
        col_dim = self._get_collection_dim(name)

        written = 0
        buf: List[IngestItem] = []

        def flush() -> None:
            nonlocal written, buf
            if not buf:
                return

            # 1) Gather texts
            texts: List[str] = []
            for it in buf:
                txt = getattr(it, it.embed_field, None)
                if not txt or not str(txt).strip():
                    raise ValueError(
                        f"Record missing text in embed_field='{it.embed_field}'. "
                        f"Available keys: {[k for k, v in it.model_dump().items() if v is not None]}"
                    )
                texts.append(str(txt))

            # 2) Encode and verify dimension
            vectors = self._encode_checked(texts, expected_dim=col_dim)

            # 3) Build payloads
            payloads: List[Dict[str, Any]] = [
                self._payload_from_item(
                    it,
                    payload_keys=payload_keys,
                    include_embed_in_payload=include_embed_in_payload,
                )
                for it in buf
            ]

            # 4) Upsert
            ids = [str(uuid4()) for _ in buf]
            self.client.upsert(
                collection_name=name,
                points=qmodels.Batch(ids=ids, vectors=vectors, payloads=payloads),
                wait=True,
            )
            written += len(buf)
            buf = []

        for it in items:
            buf.append(it)
            if len(buf) >= batch_size:
                flush()
        flush()
        return written

    # ---------------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------------
    def search(self, query: str, *, top_k: int = 5) -> List[RetrievalResult]:
        """Vector search against the current collection."""
        name = self._require_collection()
        vector = self.embedder.encode([query])[0]
        res = self.client.search(
            collection_name=name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        out: List[RetrievalResult] = []
        for p in res:
            out.append(
                RetrievalResult(
                    id=str(p.id),
                    score=float(p.score),
                    payload=dict(p.payload or {}),
                )
            )
        return out

    def search_filtered(
        self,
        query: str,
        *,
        top_k: int = 5,
        query_filter: Union[dict, qmodels.Filter],
    ) -> List[RetrievalResult]:
        """Vector search scoped by a payload filter (hybrid)."""
        name = self._require_collection()
        vector = self.embedder.encode([query])[0]
        res = self.client.search(
            collection_name=name,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        out: List[RetrievalResult] = []
        for p in res:
            out.append(
                RetrievalResult(
                    id=str(p.id),
                    score=float(p.score),
                    payload=dict(p.payload or {}),
                )
            )
        return out

    def scroll_filter(
        self,
        *,
        query_filter: Union[dict, qmodels.Filter],
        limit: int = 100,
        offset: Optional[str] = None,
    ) -> Tuple[List[RetrievalResult], Optional[str]]:
        """
        Payload-only retrieval using Qdrant scroll() with a filter.
        No similarity scores are produced; we set score=0.0 for schema consistency.
        """
        name = self._require_collection()
        points, next_offset = self.client.scroll(
            collection_name=name,
            with_payload=True,
            limit=limit,
            offset=offset,
            scroll_filter=query_filter,  # correct kwarg for qdrant_client.scroll
        )
        results: List[RetrievalResult] = [
            RetrievalResult(id=str(p.id), score=0.0, payload=dict(p.payload or {})) for p in points
        ]
        return results, next_offset

    # ---------------------------------------------------------------------
    # Mongo mirroring (for `qmem mongo`)
    # ---------------------------------------------------------------------
    def mirror_to_mongo(
        self,
        *,
        mongo_uri: str,
        mongo_db: str,
        mongo_coll: str,
        mongo_keys: Optional[Set[str]] = None,
        batch_size: int = 1000,
        max_docs: Optional[int] = None,
    ) -> int:
        """
        Mirror all (or up to max_docs) points from the current Qdrant collection into MongoDB.
        """
        name = self._require_collection()

        # Init sink (short-lived per call)
        try:
            from .mongo_sink import MongoSink  # lazy import
            sink = MongoSink(mongo_uri, mongo_db)
        except Exception as e:
            raise RuntimeError(f"Could not initialize Mongo sink: {e}") from e

        stored = 0
        next_offset: Optional[str] = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=name,
                with_payload=True,
                limit=batch_size,
                offset=next_offset,
            )

            if not points:
                break

            docs: List[Dict[str, Any]] = []
            for p in points:
                pl = dict(p.payload or {})
                doc = pl if not mongo_keys else {k: pl.get(k) for k in mongo_keys if k in pl}
                doc["_id"] = str(p.id)
                docs.append(doc)

            if docs:
                try:
                    sink.insert_many(mongo_coll, docs)
                    stored += len(docs)
                except Exception as e:
                    logger.warning("Mongo insert failed for a batch: %s", e)

            if max_docs is not None and stored >= max_docs:
                break

            if not next_offset:
                break

        try:
            sink.close()
        except Exception:
            pass

        return stored

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _require_collection(self) -> str:
        """Ensure a collection name is configured."""
        if not self.collection or not str(self.collection).strip():
            raise RuntimeError(
                "No collection configured. Pass `collection` when constructing QMem "
                "or set `default_collection` in QMemConfig."
            )
        return str(self.collection)

    def _get_collection_dim(self, name: str) -> int:
        """
        Fetch vector size (dimension) for a collection.
        Supports both single-vector and named-vectors configs.
        """
        try:
            col = self.client.get_collection(name)
        except Exception as e:
            raise RuntimeError(f"Collection '{name}' does not exist or cannot be read: {e}") from e

        vecs = getattr(getattr(col.config, "params", col.config), "vectors", None)

        # Case 1: single vector config
        try:
            if hasattr(vecs, "size"):
                return int(vecs.size)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Case 2: named vectors dict
        try:
            if isinstance(vecs, dict) and vecs:
                first = next(iter(vecs.values()))
                return int(getattr(first, "size"))
        except Exception:
            pass

        # Fallback (older clients): col.config.params.vectors.size
        try:
            return int(col.config.params.vectors.size)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover (defensive)
            raise RuntimeError(f"Could not determine vector size for '{name}': {e}") from e

    @staticmethod
    def _payload_from_item(
        it: IngestItem,
        *,
        payload_keys: Optional[Set[str]] = None,
        include_embed_in_payload: bool = False,
    ) -> Dict[str, Any]:
        """Build payload dict from IngestItem, excluding the embedded field by default."""
        d = it.model_dump(exclude_none=True)
        d.pop("embed_field", None)  # control field is not persisted

        embed_key = it.embed_field
        if not include_embed_in_payload:
            d.pop(embed_key, None)

        if payload_keys is not None:
            d = {k: v for k, v in d.items() if k in payload_keys}
        return d

    def _encode_checked(self, texts: List[str], *, expected_dim: int) -> List[List[float]]:
        """Encode texts and assert the resulting vector dimension matches the collection."""
        vectors = self.embedder.encode(texts)
        if not vectors or not vectors[0]:
            raise ValueError("Embedder returned empty vectors.")
        dim = len(vectors[0])
        if dim != expected_dim:
            raise ValueError(
                f"Vector dimension mismatch: collection expects {expected_dim}, "
                f"embedder produced {dim}."
            )
        return vectors


# =============================================================================
# Chroma implementation (local-only)
# =============================================================================
class _QMemChroma:
    """
    Local ChromaDB backend with the same public surface as the Qdrant client.
    - Persistent at cfg.chroma_path (defaults to ./.qmem/chroma)
    - No explicit payload indexes required (schema-less)
    """

    def __init__(self, cfg: QMemConfig, collection: Optional[str] = None) -> None:
        if chromadb is None:
            raise RuntimeError("ChromaDB backend requires 'chromadb'. Install with: pip install chromadb>=0.5")
        self.cfg = cfg
        self.collection = collection or cfg.default_collection
        self.embedder = build_embedder(cfg)
        path = cfg.chroma_path or ""
        self._client = chromadb.PersistentClient(path=path)
        self._col = None  # created lazily

    # ----- collection management -----
    def ensure_collection(
        self,
        *,
        create_if_missing: bool,
        distance: object = None,            # ignored by Chroma
        vector_size: Optional[int] = None,  # ignored (we trust the embedder dim)
    ) -> None:
        name = self._require_collection()
        self._col = self._client.get_or_create_collection(name)

    # ----- payload index helpers (no-ops) -----
    def create_payload_index(self, field_name: str, field_type: str) -> None:
        return

    def ensure_payload_indices(self, schema: Dict[str, str]) -> None:
        return

    # ----- ingest -----
    def ingest(
        self,
        items: Sequence[IngestItem],
        *,
        batch_size: int = 64,
        payload_keys: Optional[Set[str]] = None,
        include_embed_in_payload: bool = False,
    ) -> int:
        if not items:
            return 0

        col = self._get_collection()
        written = 0
        buf: List[IngestItem] = []

        def flush() -> None:
            nonlocal written, buf
            if not buf:
                return

            texts: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            ids: List[str] = []
            for it in buf:
                txt = getattr(it, it.embed_field, None)
                if not txt or not str(txt).strip():
                    raise ValueError(
                        f"Record missing text in embed_field='{it.embed_field}'. "
                        f"Available keys: {[k for k, v in it.model_dump().items() if v is not None]}"
                    )
                texts.append(str(txt))

                payload = it.model_dump(exclude_none=True)
                payload.pop("embed_field", None)
                if not include_embed_in_payload:
                    payload.pop(it.embed_field, None)
                if payload_keys is not None:
                    payload = {k: payload.get(k) for k in payload_keys if k in payload}

                metadatas.append(payload)
                ids.append(str(uuid4()))

            vectors = self.embedder.encode(texts)
            col.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=None)
            written += len(buf)
            buf = []

        for it in items:
            buf.append(it)
            if len(buf) >= batch_size:
                flush()
        flush()
        return written

    # ----- search -----
    def search(self, query: str, *, top_k: int = 5) -> List[RetrievalResult]:
        col = self._get_collection()
        qv = self.embedder.encode([query])[0]
        res = col.query(query_embeddings=[qv], n_results=top_k)
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if res.get("distances") else [0.0] * len(ids)

        out: List[RetrievalResult] = []
        for i, _id in enumerate(ids):
            payload = metas[i] if i < len(metas) else {}
            score = float(dists[i]) if i < len(dists) else 0.0
            out.append(RetrievalResult(id=str(_id), score=score, payload=payload))
        return out

    # ----- hybrid search with filter -----
    def search_filtered(
        self,
        query: str,
        *,
        top_k: int = 5,
        query_filter: Union[dict, object],
    ) -> List[RetrievalResult]:
        col = self._get_collection()
        qv = self.embedder.encode([query])[0]
        where = self._to_chroma_where(query_filter)
        if where:
            res = col.query(query_embeddings=[qv], n_results=top_k, where=where)
        else:
            res = col.query(query_embeddings=[qv], n_results=top_k)
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0] if res.get("distances") else [0.0] * len(ids)

        out: List[RetrievalResult] = []
        for i, _id in enumerate(ids):
            payload = metas[i] if i < len(metas) else {}
            score = float(dists[i]) if i < len(dists) else 0.0
            out.append(RetrievalResult(id=str(_id), score=score, payload=payload))
        return out

    # ----- payload-only retrieval -----
    def scroll_filter(
        self,
        *,
        query_filter: Union[dict, object],
        limit: int = 100,
        offset: Optional[str] = None,   # Chroma uses integer offsets
    ) -> Tuple[List[RetrievalResult], Optional[str]]:
        col = self._get_collection()
        where = self._to_chroma_where(query_filter)

        # Chroma expects `where` to have at least one operator; if empty, omit it.
        off = int(offset) if isinstance(offset, (int, str)) and str(offset).isdigit() else 0
        if where:
            res = col.get(where=where, limit=limit, offset=off)
        else:
            res = col.get(limit=limit, offset=off)

        ids = res.get("ids", [])
        metas = res.get("metadatas", []) or []

        results: List[RetrievalResult] = []
        for i, _id in enumerate(ids):
            payload = metas[i] if i < len(metas) else {}
            results.append(RetrievalResult(id=str(_id), score=0.0, payload=payload))

        # Next offset for simple pagination
        next_off = off + len(ids)
        return results, (str(next_off) if ids else None)

    # ----- Mongo mirror -----
    def mirror_to_mongo(
        self,
        *,
        mongo_uri: str,
        mongo_db: str,
        mongo_coll: str,
        mongo_keys: Optional[Set[str]] = None,
        batch_size: int = 1000,
        max_docs: Optional[int] = None,
    ) -> int:
        """
        Mirror all (or up to max_docs) points from the current Chroma collection into MongoDB.

        Notes:
        - Uses collection.get() with limit/offset pagination.
        - Only metadatas (payloads) are mirrored; vectors/documents are not stored in Mongo.
        """
        col = self._get_collection()

        # Init sink (short-lived per call)
        try:
            from .mongo_sink import MongoSink  # lazy import
            sink = MongoSink(mongo_uri, mongo_db)
        except Exception as e:
            raise RuntimeError(f"Could not initialize Mongo sink: {e}") from e

        stored = 0
        offset = 0

        while True:
            # No filter when mirroring everything; don't pass empty where.
            res = col.get(limit=batch_size, offset=offset)
            ids = res.get("ids", []) or []
            metas = res.get("metadatas", []) or []

            if not ids:
                break

            docs: List[Dict[str, Any]] = []
            for i, _id in enumerate(ids):
                pl = metas[i] if i < len(metas) else {}
                if mongo_keys is None:
                    doc = dict(pl) if isinstance(pl, dict) else {}
                else:
                    doc = {k: pl.get(k) for k in mongo_keys if isinstance(pl, dict) and k in pl}
                doc["_id"] = str(_id)
                docs.append(doc)

            if docs:
                try:
                    sink.insert_many(mongo_coll, docs)
                    stored += len(docs)
                except Exception as e:
                    logger.warning("Mongo insert failed for a batch: %s", e)

            offset += len(ids)
            if max_docs is not None and stored >= max_docs:
                break

        try:
            sink.close()
        except Exception:
            pass

        return stored

    # ----- helpers -----
    def _get_collection(self):
        if self._col is None:
            name = self._require_collection()
            self._col = self._client.get_or_create_collection(name)
        return self._col

    def _require_collection(self) -> str:
        if not self.collection or not str(self.collection).strip():
            raise RuntimeError(
                "No collection configured. Pass `collection` when constructing QMem "
                "or set `default_collection` in QMemConfig."
            )
        return str(self.collection)

    def _to_chroma_where(self, query_filter: Union[dict, object]) -> Dict[str, Any]:
        """
        Map a subset of qdrant-style filter JSON to Chroma where:
          - must:     AND of conditions
          - must_not: negations via $ne / $nin
          - should:   OR group via $or (best-effort)
          - match:    {"key":k,"match":{"value":v}}      -> {k: v}
          - match-any {"key":k,"match":{"any":[...]}}    -> {k: {"$in":[...]}}
          - range:    {"key":k,"range":{"gte":A,...}}    -> {k: {"$gte":A,...}}
        Unsupported: geo/has_id, complex nesting beyond simple $or/$in.
        """
        if not isinstance(query_filter, dict):
            return {}
        must = query_filter.get("must", []) or []
        must_not = query_filter.get("must_not", []) or []
        should = query_filter.get("should", []) or []

        def cond_to_where(c: dict) -> Dict[str, Any]:
            if "key" in c and "match" in c:
                key = c["key"]
                m = c["match"] or {}
                if isinstance(m, dict) and "any" in m:
                    return {key: {"$in": list(m["any"])}}
                if isinstance(m, dict) and "value" in m:
                    return {key: m["value"]}
                return {key: m}
            if "key" in c and "range" in c:
                key = c["key"]
                r = c["range"] or {}
                w: Dict[str, Any] = {}
                for a, b in (("gte", "$gte"), ("gt", "$gt"), ("lte", "$lte"), ("lt", "$lt")):
                    if a in r:
                        w[b] = r[a]
                return {key: w} if w else {}
            return {}

        where: Dict[str, Any] = {}
        for c in must:
            where.update(cond_to_where(c))

        for c in must_not:
            d = cond_to_where(c)
            for k, v in d.items():
                if isinstance(v, dict) and "$in" in v:
                    where[k] = {"$nin": v["$in"]}
                else:
                    where[k] = {"$ne": v}

        if should:
            ors = [cond_to_where(c) for c in should if cond_to_where(c)]
            if ors:
                where = {"$or": [where] + ors} if where else {"$or": ors}

        return where


# =============================================================================
# Public façade that selects backend by cfg.vector_store
# =============================================================================
class QMem:
    """
    Facade over Qdrant (cloud) and Chroma (local) backends with the same methods:
      - ensure_collection
      - create_payload_index / ensure_payload_indices
      - ingest
      - search / search_filtered
      - scroll_filter
      - mirror_to_mongo  (supported for both backends)
    """

    def __init__(self, cfg: QMemConfig, collection: Optional[str] = None) -> None:
        self._backend = (getattr(cfg, "vector_store", "qdrant") or "qdrant").lower()
        if self._backend == "chroma":
            self._impl = _QMemChroma(cfg, collection)
        else:
            self._impl = _QMemQdrant(cfg, collection)

    # Delegate all methods 1:1

    def ensure_collection(self, **kw) -> None:
        return self._impl.ensure_collection(**kw)

    def create_payload_index(self, *a, **kw) -> None:
        return self._impl.create_payload_index(*a, **kw)

    def ensure_payload_indices(self, *a, **kw) -> None:
        return self._impl.ensure_payload_indices(*a, **kw)

    def ingest(self, *a, **kw) -> int:
        return self._impl.ingest(*a, **kw)

    def search(self, *a, **kw) -> List[RetrievalResult]:
        return self._impl.search(*a, **kw)

    def search_filtered(self, *a, **kw) -> List[RetrievalResult]:
        return self._impl.search_filtered(*a, **kw)

    def scroll_filter(self, *a, **kw) -> Tuple[List[RetrievalResult], Optional[str]]:
        return self._impl.scroll_filter(*a, **kw)

    def mirror_to_mongo(self, *a, **kw) -> int:
        return self._impl.mirror_to_mongo(*a, **kw)