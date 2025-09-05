from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Protocol, Sequence, runtime_checkable

__all__ = [
    "Embedder",
    "OpenAIEmbedder",
    "MiniLMEmbedder",
    "build_embedder",
]

# ---------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Embedding interface used by QMem."""

    dim: int

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Return one vector per input string (len(vector) == self.dim)."""
        ...


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def _chunks(seq: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _l2_normalize(vec: List[float]) -> List[float]:
    s = sum(x * x for x in vec) ** 0.5
    if s == 0.0:
        return vec
    return [x / s for x in vec]


def _ensure_dim(vec: List[float], wanted: int) -> List[float]:
    """Trim or pad so we always return the configured dimension."""
    n = len(vec)
    if n == wanted:
        return vec
    if n > wanted:
        return vec[:wanted]
    return vec + [0.0] * (wanted - n)


def _to_list(v: Any) -> List[float]:
    """Best-effort convert embedding vector to a plain Python list[float]."""
    # NumPy array / torch tensor / anything with .tolist()
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, tuple):
        v = list(v)
    if not isinstance(v, list):
        raise RuntimeError(f"Unexpected embedding vector type: {type(v)}")
    return [float(x) for x in v]


# ---------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------


@dataclass
class _OpenAIConfig:
    model: str
    api_key: str
    dim: int
    batch_size: int = 128
    normalize: bool = False


class OpenAIEmbedder:
    """
    OpenAI embedding backend.

    Notes:
    - "text-embedding-3-small" => 1536 dims
    - "text-embedding-3-large" => 3072 dims
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        dim: int,
        *,
        batch_size: int = 128,
        normalize: bool = False,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAI embeddings")

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI backend requires the 'openai' package. Install with:\n"
                "  pip install openai"
            ) from e

        self._cfg = _OpenAIConfig(
            model=model, api_key=api_key, dim=dim, batch_size=batch_size, normalize=normalize
        )
        self.client = OpenAI(api_key=api_key)
        self.dim = dim

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        out: List[List[float]] = []
        for batch in _chunks(texts, self._cfg.batch_size):
            resp = self.client.embeddings.create(model=self._cfg.model, input=list(batch))
            for d in resp.data:
                vec = list(d.embedding)
                if self._cfg.normalize:
                    vec = _l2_normalize(vec)
                out.append(_ensure_dim(vec, self._cfg.dim))
        return out


# ---------------------------------------------------------------------
# MiniLM (Hugging Face Inference API — NO DOWNLOADS)
# ---------------------------------------------------------------------


@dataclass
class _HFAPIConfig:
    model: str
    token: str
    dim: int
    batch_size: int = 256
    normalize: bool = False


class MiniLMEmbedder:
    """
    Hosted MiniLM via Hugging Face Inference API.

    - No local model download.
    - Works with huggingface_hub versions that have either:
        * InferenceClient.embeddings(...)
        * InferenceClient.feature_extraction(...)
    - Default model: sentence-transformers/all-MiniLM-L6-v2 (384 dims)
    """

    def __init__(
        self,
        model: str,
        token: str,
        dim: int,
        *,
        batch_size: int = 256,
        normalize: bool = False,
    ) -> None:
        if not token:
            raise ValueError("Hugging Face API key is required for MiniLM (HF) embeddings")
        try:
            from huggingface_hub import InferenceClient  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "HF backend requires 'huggingface_hub'. Install/upgrade with:\n"
                "  pip install -U huggingface_hub>=0.24"
            ) from e

        self._cfg = _HFAPIConfig(
            model=model, token=token, dim=dim, batch_size=batch_size, normalize=normalize
        )
        self.client = InferenceClient(token=token)
        self.dim = dim

        # Detect available API
        self._has_embeddings = hasattr(self.client, "embeddings")
        self._has_feature_extraction = hasattr(self.client, "feature_extraction")
        if not (self._has_embeddings or self._has_feature_extraction):
            raise RuntimeError(
                "Your huggingface_hub client has neither `.embeddings` nor `.feature_extraction`.\n"
                "Please upgrade: pip install -U huggingface_hub>=0.24"
            )

    def _normalize_response(self, resp: Any) -> List[List[float]]:
        """
        Normalize various HF outputs to List[List[float]].

        Accepted shapes/types:
          - numpy.ndarray or torch.Tensor: (B, D) or (D,)
          - List[List[float]] or List[float]
          - Dict with keys: 'embeddings' | 'embedding' | 'vectors' | 'data'
          - Objects with attribute .embeddings or .data
        """
        # Attribute-style access (some clients return objects with .embeddings / .data)
        for attr in ("embeddings", "data"):
            if hasattr(resp, attr):
                resp = getattr(resp, attr)

        # Dict wrappers
        if isinstance(resp, dict):
            for key in ("embeddings", "embedding", "vectors", "data"):
                if key in resp:
                    resp = resp[key]
                    break

        # NumPy / Torch -> tolist()
        if hasattr(resp, "tolist"):
            resp = resp.tolist()

        # Single vector -> wrap
        if isinstance(resp, list) and resp and isinstance(resp[0], (int, float)):
            resp = [resp]

        if not isinstance(resp, list):
            raise RuntimeError(f"Unexpected HF response type: {type(resp)}")

        return [_to_list(v) for v in resp]

    def encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        out: List[List[float]] = []
        for batch in _chunks(texts, self._cfg.batch_size):
            if self._has_embeddings:
                # Newer huggingface_hub signature
                resp = self.client.embeddings(model=self._cfg.model, inputs=list(batch))
            else:
                # Older huggingface_hub signature uses `text=...`
                resp = self.client.feature_extraction(text=list(batch), model=self._cfg.model)

            embs = self._normalize_response(resp)
            for v in embs:
                vec = v
                if self._cfg.normalize:
                    vec = _l2_normalize(vec)
                out.append(_ensure_dim(vec, self._cfg.dim))

        return out


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------


def build_embedder(cfg) -> Embedder:
    """
    Construct the appropriate embedder from a QMemConfig-like object.

    Expected fields on cfg:
        - embed_provider in {"openai", "minilm"}
        - embed_model: str
        - embed_dim: int
        - openai_api_key: Optional[str] (if provider == "openai")
        - hf_api_key: Optional[str] (if provider == "minilm")
    """
    if cfg.embed_provider == "openai":
        return OpenAIEmbedder(cfg.embed_model, cfg.openai_api_key or "", cfg.embed_dim)
    if cfg.embed_provider == "minilm":
        # Use hosted MiniLM via Hugging Face Inference API (no downloads)
        return MiniLMEmbedder(cfg.embed_model, getattr(cfg, "hf_api_key", "") or "", cfg.embed_dim)
    raise ValueError(f"Unsupported embed provider: {cfg.embed_provider!r}")
