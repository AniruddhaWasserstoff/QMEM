from __future__ import annotations

"""
QMem-backed Letta tools that replace archival memory.
- archival_memory_insert  -> stores text in QMem (Qdrant/Chroma)
- archival_memory_search  -> retrieves text from QMem via semantic search
"""

from typing import List, ClassVar
from pydantic import BaseModel, Field

# Handle BaseTool import across letta-client versions
try:
    from letta_client.client import BaseTool
except Exception:  # older/newer path fallback
    from letta_client.tools import BaseTool  # type: ignore

# Use QMem low-level API (already inside this repo)
from qmem.api import create as qmem_create
from qmem.api import ingest as qmem_ingest
from qmem.api import retrieve as qmem_retrieve

# --------------------------
# Config
# --------------------------
QMEM_COLLECTION = "letta_agent_memory"
DEFAULT_TOP_K = 5


def _ensure_qmem_collection() -> None:
    """
    Ensure the QMem collection exists.
    qmem.api.create() is idempotent (no-op if already present).
    Uses embed_dim from .qmem/config.toml by default.
    """
    qmem_create(QMEM_COLLECTION)


# --------------------------
# Arg schemas
# --------------------------
class MemoryInsertArgs(BaseModel):
    content: str = Field(..., description="Content to store in long-term memory")


class MemorySearchArgs(BaseModel):
    query: str = Field(..., description="Semantic search query")
    k: int = Field(DEFAULT_TOP_K, description="Top-K results to return")


# --------------------------
# Tools
# --------------------------
class QMemInsertTool(BaseTool):
    """Replace Letta's archival_memory_insert with QMem ingestion."""
    name: ClassVar[str] = "archival_memory_insert"
    description: ClassVar[str] = "Store long-term memory via QMem (Qdrant/Chroma backend)"
    args_schema: ClassVar[type[BaseModel]] = MemoryInsertArgs

    def run(self, content: str) -> str:
        _ensure_qmem_collection()
        # Store text under payload key "text"
        qmem_ingest(
            QMEM_COLLECTION,
            records=[{"text": content}],
            embed_field="text",
            include_embed_in_payload=True,
        )
        return "stored"


class QMemSearchTool(BaseTool):
    """Replace Letta's archival_memory_search with QMem semantic search."""
    name: ClassVar[str] = "archival_memory_search"
    description: ClassVar[str] = "Search long-term memory via QMem (Qdrant/Chroma backend)"
    args_schema: ClassVar[type[BaseModel]] = MemorySearchArgs

    def run(self, query: str, k: int = DEFAULT_TOP_K) -> str:
        _ensure_qmem_collection()
        results = qmem_retrieve(QMEM_COLLECTION, query, k=k)

        # qmem.api.retrieve returns a list[RetrievalResult] with .payload dicts
        lines: List[str] = []
        for r in results:
            payload = getattr(r, "payload", {}) or {}
            txt = payload.get("text")
            if txt:
                lines.append(str(txt))

        return "\n".join(lines)
