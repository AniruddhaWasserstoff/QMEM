from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
import tomlkit as toml

__all__ = ["QMemConfig", "CONFIG_DIR", "CONFIG_PATH", "FILTERS_DIR"]

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

CONFIG_DIR = Path(".qmem")
CONFIG_PATH = CONFIG_DIR / "config.toml"
FILTERS_DIR = CONFIG_DIR / "filters"  # where saved filter JSONs live

# Environment variable names (env > file when loading)
_ENV = {
    "qdrant_url": "QMEM_QDRANT_URL",
    "qdrant_api_key": "QMEM_QDRANT_API_KEY",
    "openai_api_key": "QMEM_OPENAI_API_KEY",
    "hf_api_key": "QMEM_HF_API_KEY",
    "gemini_api_key": "QMEM_GEMINI_API_KEY",
    "voyage_api_key": "QMEM_VOYAGE_API_KEY",
    "embed_provider": "QMEM_EMBED_PROVIDER",
    "embed_model": "QMEM_EMBED_MODEL",
    "embed_dim": "QMEM_EMBED_DIM",
    "default_collection": "QMEM_DEFAULT_COLLECTION",
    # NEW for backend selection:
    "vector_store": "QMEM_VECTOR_STORE",  # qdrant | chroma
    "chroma_path": "QMEM_CHROMA_PATH",    # path for local Chroma persistence
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _mask(value: Optional[str], keep: int = 4) -> str:
    """Return a redacted representation of a secret for safe display."""
    if not value:
        return ""
    v = str(value)
    if len(v) <= keep:
        return "*" * len(v)
    return f"{v[:keep]}…{'*' * 4}"


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class QMemConfig(BaseModel):
    """
    App configuration (service endpoints, keys, embedding settings).

    Environment variable overrides (if set) take precedence when loading:
      - QMEM_QDRANT_URL
      - QMEM_QDRANT_API_KEY
      - QMEM_OPENAI_API_KEY
      - QMEM_HF_API_KEY
      - QMEM_GEMINI_API_KEY
      - QMEM_VOYAGE_API_KEY
      - QMEM_EMBED_PROVIDER  (openai|minilm|gemini|voyage)
      - QMEM_EMBED_MODEL
      - QMEM_EMBED_DIM
      - QMEM_DEFAULT_COLLECTION
      - QMEM_VECTOR_STORE     (qdrant|chroma)
      - QMEM_CHROMA_PATH
    """

    # Backend selection
    vector_store: Literal["qdrant", "chroma"] = Field(
        default="qdrant", description="Vector store backend (qdrant | chroma)"
    )
    chroma_path: Optional[str] = Field(
        default=None,
        description="Local path for Chroma persistent client (defaults to ./.qmem/chroma)",
    )

    # Services (Qdrant fields are required only when using Qdrant)
    qdrant_url: str = Field(
        default="http://localhost:6333", description="Qdrant endpoint URL"
    )
    qdrant_api_key: str = Field(
        default="", description="Qdrant API key"
    )

    # Embedding provider keys
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key (if using OpenAI embeddings)"
    )
    hf_api_key: Optional[str] = Field(
        default=None,
        description="Hugging Face API key (if using hosted MiniLM via HF Inference API)",
    )
    gemini_api_key: Optional[str] = Field(
        default=None, description="Google Gemini API key (if using Gemini embeddings)"
    )
    voyage_api_key: Optional[str] = Field(
        default=None, description="Voyage AI API key (if using Voyage embeddings)"
    )

    # Embeddings
    embed_provider: Literal["openai", "minilm", "gemini", "voyage"]
    embed_model: str
    embed_dim: int

    # Optional default collection
    default_collection: Optional[str] = None

    # -----------------------
    # Safe representations
    # -----------------------

    def __repr__(self) -> str:  # pragma: no cover - representation only
        return (
            "QMemConfig("
            f"vector_store={self.vector_store!r}, "
            f"chroma_path={self.chroma_path!r}, "
            f"qdrant_url={self.qdrant_url!r}, "
            f"qdrant_api_key={_mask(self.qdrant_api_key)!r}, "
            f"openai_api_key={_mask(self.openai_api_key)!r}, "
            f"hf_api_key={_mask(self.hf_api_key)!r}, "
            f"gemini_api_key={_mask(self.gemini_api_key)!r}, "
            f"voyage_api_key={_mask(self.voyage_api_key)!r}, "
            f"embed_provider={self.embed_provider!r}, "
            f"embed_model={self.embed_model!r}, "
            f"embed_dim={self.embed_dim!r}, "
            f"default_collection={self.default_collection!r}"
            ")"
        )

    __str__ = __repr__  # keep secrets masked in str()

    def masked_dict(self) -> dict:
        """Return a dict where secrets are redacted (for safe logging)."""
        d = self.model_dump()
        d["qdrant_api_key"] = _mask(self.qdrant_api_key)
        d["openai_api_key"] = _mask(self.openai_api_key)
        d["hf_api_key"] = _mask(self.hf_api_key)
        d["gemini_api_key"] = _mask(self.gemini_api_key)
        d["voyage_api_key"] = _mask(self.voyage_api_key)
        return d

    # -----------------------
    # Persistence
    # -----------------------

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "QMemConfig":
        """
        Load config from TOML, then apply env var overrides.

        Raises:
            FileNotFoundError: if the file doesn't exist AND required envs are missing.
        """
        data: dict = {}
        if path.exists():
            # toml.parse returns a TOMLDocument (dict-like). That's fine for our merge.
            data = dict(toml.parse(path.read_text(encoding="utf-8")))

        # Apply environment overrides if present (env > file; ignore empty strings)
        merged = {**data}
        for field, env_name in _ENV.items():
            v = os.getenv(env_name)
            if v is not None and v != "":
                merged[field] = v

        # Provide defaults for new fields if missing
        vstore = merged.get("vector_store")
        if vstore not in ("qdrant", "chroma"):
            merged["vector_store"] = "qdrant"

        if not merged.get("chroma_path"):
            merged["chroma_path"] = str((CONFIG_DIR / "chroma").resolve())

        # If file missing AND required fields absent, guide user to init.
        # Embedding settings are always required.
        need_embed = any(k not in merged for k in ("embed_provider", "embed_model", "embed_dim"))
        # Qdrant creds are required only when vector_store == "qdrant"
        need_qdrant = (
            merged.get("vector_store") == "qdrant"
            and any(k not in merged for k in ("qdrant_url", "qdrant_api_key"))
        )
        if not path.exists() and (need_embed or need_qdrant):
            raise FileNotFoundError(f"Config not found at {path}. Run `qmem init` in this folder.")

        # Normalize empty strings to None for optional fields (e.g., API keys)
        cleaned = {k: (v if v != "" else None) for k, v in merged.items()}

        # Ensure qdrant_url/api_key have safe defaults to keep client code simple
        if cleaned.get("qdrant_url") is None:
            cleaned["qdrant_url"] = "http://localhost:6333"
        if cleaned.get("qdrant_api_key") is None:
            cleaned["qdrant_api_key"] = ""

        # Pydantic will coerce embed_dim to int if it's a str from env.
        return cls(**cleaned)

    def save(self, path: Path = CONFIG_PATH) -> None:
        """
        Save config to TOML and set file permissions to 600 (owner read/write only).
        Secrets are stored as-is, but we never print them.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Default chroma path if not set
        chroma_path = self.chroma_path or str((CONFIG_DIR / "chroma").resolve())

        doc = toml.document()

        # Backend
        doc.add("vector_store", self.vector_store)
        doc.add("chroma_path", chroma_path)

        # Services
        doc.add("qdrant_url", self.qdrant_url or "http://localhost:6333")
        doc.add("qdrant_api_key", self.qdrant_api_key or "")

        # Keys
        doc.add("openai_api_key", self.openai_api_key or "")
        doc.add("hf_api_key", self.hf_api_key or "")
        doc.add("gemini_api_key", self.gemini_api_key or "")
        doc.add("voyage_api_key", self.voyage_api_key or "")

        # Embeddings
        doc.add("embed_provider", self.embed_provider)
        doc.add("embed_model", self.embed_model)
        doc.add("embed_dim", int(self.embed_dim))

        # Defaults
        doc.add("default_collection", self.default_collection or "")

        path.write_text(toml.dumps(doc), encoding="utf-8")

        # Restrict permissions: -rw-------
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            # Non-POSIX or insufficient permissions—silently ignore.
            pass