from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

# Which field in each record is allowed to be embedded
EmbedField = Literal["query", "response", "sql_query", "doc_id"]


class IngestItem(BaseModel):
    """
    Source record to ingest & embed.

    Notes:
      - The actual embedded text is taken from `embed_field`.
      - Validation is intentionally lenient here; strict checks happen in QMem.ingest().
      - CLI/API may place unknown input keys under `extra` before model creation.
    """

    # Pydantic v2: ignore unknown fields (extras are handled by the caller)
    model_config = ConfigDict(extra="ignore")

    # Optional text fields your records may contain
    query: Optional[str] = None
    response: Optional[str] = None
    sql_query: Optional[str] = None
    doc_id: Optional[str] = None

    # Arbitrary graph/payload metadata
    graph: Optional[Dict[str, Any]] = None
    tags: Optional[Any] = None
    extra: Optional[Dict[str, Any]] = None

    # Which of the above fields to embed
    embed_field: EmbedField = "response"

    @field_validator("embed_field")
    @classmethod
    def _ensure_embed_present(cls, v: str, info: ValidationInfo) -> str:
        """
        Soft-check that the chosen embed field exists on the instance.
        We do not fail here; ingest() will enforce non-empty text later.
        """
        _ = (getattr(info, "data", {}) or {}).get(v)
        return v


class RetrievalResult(BaseModel):
    """Typed wrapper for search results."""
    id: str
    score: float
    payload: Dict[str, Any]