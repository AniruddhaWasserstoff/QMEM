# src/qmem/qmem_tools.py
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Union

from pydantic import BaseModel, Field

# --- Import Letta tool base (supports both modern and older client layouts) ---
try:
    from letta_client.client import BaseTool
except Exception:
    try:
        from letta_client.tools import BaseTool  # type: ignore
    except Exception as e:
        raise RuntimeError("Could not import Letta BaseTool from letta_client") from e

# --- QMem APIs & config ---
from qmem.api import create as qmem_create, ingest as qmem_ingest, retrieve as qmem_retrieve
from qmem.config import QMemConfig, CONFIG_PATH

DEFAULT_TOP_K = 5
MAX_INSERT_CHARS = 20_000


# ---------------------------
# Internals
# ---------------------------
def _load_collection() -> str:
    """
    Resolve the active collection:
    - If default_collection exists in config, use it.
    - Otherwise, persist the fallback ('letta_agent_memory') into config for stability.
    """
    cfg = QMemConfig.load(CONFIG_PATH)
    if getattr(cfg, "default_collection", None) and str(cfg.default_collection).strip():
        return cfg.default_collection

    cfg.default_collection = "letta_agent_memory"
    cfg.save(CONFIG_PATH)
    return cfg.default_collection


def _ensure_qmem_collection() -> str:
    """
    Ensure the active collection exists (idempotent create) and return its name.
    """
    coll = _load_collection()
    qmem_create(coll)
    return coll


def _coerce_args(args_or_kwargs: Union[BaseModel, Dict[str, Any]], model: type[BaseModel]) -> BaseModel:
    """
    Letta may pass a BaseModel instance, kwargs dict, or a singleton list containing a kwargs dict.
    Normalize into a proper Pydantic model for our tool.
    """
    if isinstance(args_or_kwargs, BaseModel):
        return args_or_kwargs
    if isinstance(args_or_kwargs, dict):
        return model(**args_or_kwargs)
    if isinstance(args_or_kwargs, (list, tuple)) and len(args_or_kwargs) == 1 and isinstance(args_or_kwargs[0], dict):
        return model(**args_or_kwargs[0])
    raise TypeError(f"Unexpected args payload: {type(args_or_kwargs)!r}")


# ---------------------------
# Pydantic schemas
# ---------------------------
class MemoryInsertArgs(BaseModel):
    content: str = Field(..., description="Content to store in long-term memory")


class MemorySearchArgs(BaseModel):
    query: str = Field(..., description="Semantic search query")
    k: int = Field(DEFAULT_TOP_K, description="Top-K results to return")


# ---------------------------
# Tools
# ---------------------------
class QMemInsertTool(BaseTool):
    """
    Primary archival memory insert tool backed by QMem.
    """
    name: ClassVar[str] = "archival_memory_insert"
    description: ClassVar[str] = "Store long-term memory via QMem"
    args_schema: ClassVar[type[BaseModel]] = MemoryInsertArgs

    def run(self, *tool_args: Any, **tool_kwargs: Any) -> str:
        # Normalize args
        args_raw: Union[BaseModel, Dict[str, Any]] = tool_kwargs or (tool_args[0] if tool_args else {})
        # Accept 'text' as an alias for 'content' to be friendly to generic agents
        if isinstance(args_raw, dict) and "content" not in args_raw and "text" in args_raw:
            args_raw = {**args_raw, "content": args_raw.get("text", "")}
        args = _coerce_args(args_raw, MemoryInsertArgs)

        # Ensure collection exists
        coll = _ensure_qmem_collection()
        cfg = QMemConfig.load(CONFIG_PATH)

        # Validate + truncate
        content = (args.content or "").strip()
        if not content:
            return "ignored: empty content"
        if len(content) > MAX_INSERT_CHARS:
            content = content[:MAX_INSERT_CHARS]

        # Optional lightweight tagging (handy for later filters)
        tags: List[str] = []
        lc = content.lower()
        if "my name is" in lc or "i am " in lc:
            tags.append("user_profile")
        if "college" in lc or "university" in lc:
            tags.append("education")

        record: Dict[str, Any] = {"text": content}
        if tags:
            record["tags"] = tags

        try:
            qmem_ingest(
                coll,
                records=[record],
                embed_field="text",
                include_embed_in_payload=True,
            )
            backend = getattr(cfg, "vector_store", "qdrant")
            return f"stored to collection='{coll}' on backend='{backend}'"
        except Exception as e:
            return f"error: {e}"


class QMemSearchTool(BaseTool):
    """
    Primary archival memory search tool backed by QMem.
    """
    name: ClassVar[str] = "archival_memory_search"
    description: ClassVar[str] = "Search long-term memory via QMem"
    args_schema: ClassVar[type[BaseModel]] = MemorySearchArgs

    def run(self, *tool_args: Any, **tool_kwargs: Any) -> str:
        # Normalize args
        args_raw: Union[BaseModel, Dict[str, Any]] = tool_kwargs or (tool_args[0] if tool_args else {})
        args = _coerce_args(args_raw, MemorySearchArgs)

        coll = _ensure_qmem_collection()
        query = (args.query or "").strip()
        if not query:
            return ""

        try:
            k = max(1, int(args.k or DEFAULT_TOP_K))
        except Exception:
            k = DEFAULT_TOP_K

        try:
            results = qmem_retrieve(coll, query, k=k)
        except Exception as e:
            return f"error: {e}"

        lines: List[str] = []
        for r in results:
            payload = getattr(r, "payload", {}) or {}
            txt = payload.get("text")
            if txt:
                lines.append(str(txt))
        return "\n".join(lines)


# --- Aliases (kept for backward compat but NOT exported/registered by default) ---
class QMemInsertAliasAsMemoryInsert(QMemInsertTool):
    name: ClassVar[str] = "memory_insert"
    description: ClassVar[str] = "Alias: memory_insert → QMem ingestion"


class QMemSearchAliasAsMemorySearch(QMemSearchTool):
    name: ClassVar[str] = "memory_search"
    description: ClassVar[str] = "Alias: memory_search → QMem retrieval"


__all__ = [
    "QMemInsertTool",
    "QMemSearchTool",
    "register_with_letta",
]


# ---------------------------
# Convenience: register tools with a Letta client
# ---------------------------
def register_with_letta(client: Any, *, include_aliases: bool = False) -> None:
    """
    Idempotently register QMem tools with a Letta client.
    By default, ONLY the canonical archival tools are registered.
    Set include_aliases=True to additionally register legacy 'memory_*' aliases.

    Example:
        from letta_client import Letta
        from qmem.qmem_tools import register_with_letta

        lc = Letta(base_url=os.environ.get("LETTA_BASE_URL", "http://localhost:8283"),
                   token=os.environ.get("LETTA_TOKEN"))
        register_with_letta(lc, include_aliases=False)
    """
    # Primary tools
    try:
        client.tools.add(tool=QMemInsertTool())
    except Exception:
        pass
    try:
        client.tools.add(tool=QMemSearchTool())
    except Exception:
        pass

    # Optional aliases for generic/legacy agent templates
    if include_aliases:
        try:
            client.tools.add(tool=QMemInsertAliasAsMemoryInsert())
        except Exception:
            pass
        try:
            client.tools.add(tool=QMemSearchAliasAsMemorySearch())
        except Exception:
            pass