from __future__ import annotations

import logging as log
from typing import Any, Dict, List, Optional, Sequence, Set
from uuid import uuid4

from qdrant_client import QdrantClient, models as qmodels

from .config import QMemConfig
from .embedder import build_embedder
from .schemas import IngestItem, RetrievalResult

logger = log.getLogger(__name__)


class QMem:
    """Minimal client for ingest & retrieval against a single Qdrant collection."""

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

        Args:
            create_if_missing: If True, creates the collection when missing.
            distance: Distance metric to configure when creating.
            vector_size: Vector dimension for the collection (defaults to embedder.dim).
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

        Args:
            items: Records to ingest.
            batch_size: Upsert in batches of this size.
            payload_keys: If provided, only these keys are kept in payload.
            include_embed_in_payload: If True, also store the embedded field in payload.

        Returns:
            Number of items written.

        Raises:
            RuntimeError: If collection is unset or missing.
            ValueError: If a record lacks text in the chosen embed_field or dims mismatch.
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
        """
        Vector search against the current collection.

        Args:
            query: Natural language or keyword query.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult objects with score and payload.
        """
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

        # qdrant_client >= 1.7 typically: col.config.params.vectors
        vecs = getattr(getattr(col.config, "params", col.config), "vectors", None)

        # Case 1: single vector config
        try:
            if hasattr(vecs, "size"):
                return int(vecs.size)  # type: ignore[attr-defined]
        except Exception:
            pass

        # Case 2: named vectors dict
        try:
            # vecs may be a dict-like: {"text": VectorParams(size=..., ...), ...}
            if isinstance(vecs, dict) and vecs:
                # pick the first vector space's size
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
        """
        Build payload dict from IngestItem, excluding the embedded field by default.

        - payload_keys=None means "all available (except embedded)".
        - include_embed_in_payload=False ensures the embedded text is never stored.
        """
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
