from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


class IngestItem(BaseModel):
    """
    Source record to ingest & embed.

    Notes:
      - The actual embedded text is taken from `embed_field`.
      - `embed_field` can be ANY key present on the record (no fixed whitelist).
      - Unknown keys are allowed and pass through into the payload (flat payload behavior).
    """

    # Pydantic v2: allow unknown fields to pass through onto the model
    model_config = ConfigDict(extra="allow")

    # Optional text fields your records may contain (common conveniences)
    query: Optional[str] = None
    response: Optional[str] = None
    sql_query: Optional[str] = None
    doc_id: Optional[str] = None

    # Arbitrary graph/payload metadata
    graph: Optional[Dict[str, Any]] = None
    tags: Optional[Any] = None
    extra: Optional[Dict[str, Any]] = None  # harmless; unused now

    # Which field to embed (NOW: any field name allowed)
    embed_field: str = "response"

    @field_validator("embed_field")
    @classmethod
    def _ensure_embed_present(cls, v: str, info: ValidationInfo) -> str:
        """
        Soft-check that the chosen embed field exists on the instance.
        We do not fail here; client.ingest() will enforce non-empty text later.
        """
        _ = (getattr(info, "data", {}) or {}).get(v)
        return v


class RetrievalResult(BaseModel):
    """Typed wrapper for search results."""
    id: str
    score: float
    payload: Mapping[str, Any]
