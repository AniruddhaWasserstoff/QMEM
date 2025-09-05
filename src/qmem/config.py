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
FILTERS_DIR = CONFIG_DIR / "filters"  # new: where saved filter JSONs live

# Environment variable names
_ENV = {
    "qdrant_url": "QMEM_QDRANT_URL",
    "qdrant_api_key": "QMEM_QDRANT_API_KEY",
    "openai_api_key": "QMEM_OPENAI_API_KEY",
    "hf_api_key": "QMEM_HF_API_KEY",
    "embed_provider": "QMEM_EMBED_PROVIDER",
    "embed_model": "QMEM_EMBED_MODEL",
    "embed_dim": "QMEM_EMBED_DIM",
    "default_collection": "QMEM_DEFAULT_COLLECTION",
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
      - QMEM_EMBED_PROVIDER  (openai|minilm)
      - QMEM_EMBED_MODEL
      - QMEM_EMBED_DIM
      - QMEM_DEFAULT_COLLECTION
    """

    # Services
    qdrant_url: str = Field(..., description="Qdrant endpoint URL")
    qdrant_api_key: str = Field(..., description="Qdrant API key")
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key (if using OpenAI embeddings)"
    )
    hf_api_key: Optional[str] = Field(
        default=None,
        description="Hugging Face API key (if using hosted MiniLM via HF Inference API)",
    )

    # Embeddings
    embed_provider: Literal["openai", "minilm"]
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
            f"qdrant_url={self.qdrant_url!r}, "
            f"qdrant_api_key={_mask(self.qdrant_api_key)!r}, "
            f"openai_api_key={_mask(self.openai_api_key)!r}, "
            f"hf_api_key={_mask(self.hf_api_key)!r}, "
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

        # If file missing AND required fields absent, guide user to init.
        required = ("qdrant_url", "qdrant_api_key", "embed_provider", "embed_model", "embed_dim")
        if not path.exists() and any(k not in merged for k in required):
            raise FileNotFoundError(f"Config not found at {path}. Run `qmem init` in this folder.")

        # Normalize empty strings to None for optional fields (e.g., API keys)
        cleaned = {k: (v if v != "" else None) for k, v in merged.items()}

        # Pydantic will coerce embed_dim to int if it's a str from env.
        return cls(**cleaned)

    def save(self, path: Path = CONFIG_PATH) -> None:
        """
        Save config to TOML and set file permissions to 600 (owner read/write only).
        Secrets are stored as-is, but we never print them.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        doc = toml.document()
        doc.add("qdrant_url", self.qdrant_url)
        doc.add("qdrant_api_key", self.qdrant_api_key)
        doc.add("openai_api_key", self.openai_api_key or "")
        doc.add("hf_api_key", self.hf_api_key or "")
        doc.add("embed_provider", self.embed_provider)
        doc.add("embed_model", self.embed_model)
        doc.add("embed_dim", int(self.embed_dim))
        doc.add("default_collection", self.default_collection or "")

        path.write_text(toml.dumps(doc), encoding="utf-8")

        # Restrict permissions: -rw-------
        try:
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            # Non-POSIX or insufficient permissions—silently ignore.
            pass
