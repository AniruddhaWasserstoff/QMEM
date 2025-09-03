from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import questionary as qs
import typer
from qdrant_client import models as qmodels
from rich import box
from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from .client import QMem
from .config import CONFIG_PATH, QMemConfig
from .schemas import IngestItem

app = typer.Typer(help="qmem — simple Qdrant wrapper for ingestion & retrieval")
console = Console()

# -----------------------------
# Utilities
# -----------------------------

_DISTANCE_MAP: Dict[str, qmodels.Distance] = {
    "cosine": qmodels.Distance.COSINE,
    "dot": qmodels.Distance.DOT,
    "euclid": qmodels.Distance.EUCLID,
}


def _fail(msg: str, code: int = 1) -> None:
    console.print(f"[red]{msg}[/red]")
    raise typer.Exit(code=code)


def _read_records(path: Path) -> List[dict]:
    """Load records from a .json or .jsonl file with clear errors."""
    if not path.exists():
        _fail(f"No such file: {path}")

    try:
        if path.suffix.lower() == ".jsonl":
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return [json.loads(ln) for ln in lines]
        # .json
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, list) else [obj]
    except json.JSONDecodeError as e:
        _fail(f"Invalid JSON in {path}: {e}")
    except Exception as e:  # defensive
        _fail(f"Failed to read {path}: {e}")
    return []  # unreachable


def _ensure_collection_exists(q: QMem, name: str) -> None:
    try:
        q.client.get_collection(name)
    except Exception:
        _fail(f"No such collection: {name}")


def _parse_payload_keys(raw: str) -> Optional[Set[str]]:
    """Parse comma-separated payload keys -> set[str] or None for default behavior."""
    raw = (raw or "").strip()
    if not raw:
        return None
    return {k.strip() for k in raw.split(",") if k.strip()}


def _collect_string_fields(records: Sequence[dict]) -> List[str]:
    """
    Discover keys that have at least one non-empty string value across records.
    Ordered by (frequency desc, then name).
    """
    freq: Counter[str] = Counter()
    for d in records:
        for k, v in d.items():
            if isinstance(v, str) and v.strip():
                freq[k] += 1
    return [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]


def _truncate(s: object, limit: int = 80) -> str:
    text = "" if s is None else str(s)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# -----------------------------
# init
# -----------------------------

@app.command("init", help="Configure keys + embedding model (saved to ./.qmem/config.toml)")
def init_cmd() -> None:
    cfg_path = CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("qmem init — set keys and choose embedding model", style="bold")
    openai_key = Prompt.ask(
        "OpenAI API key (leave empty if using MiniLM)",
        default="",
        show_default=False,
        password=True,
    )

    qdrant_url = Prompt.ask("Qdrant URL (e.g., https://xxxx.cloud.qdrant.io)")
    qdrant_key = Prompt.ask("Qdrant API key", password=True)

    model_choice = qs.select(
        "Choose embedding model:",
        choices=[
            "text-embedding-3-small (OpenAI, 1536)",
            "text-embedding-3-large (OpenAI, 3072)",
            "MiniLM sentence-transformer (HF, 384)",
        ],
        pointer="➤",
        use_shortcuts=False,
    ).ask()

    if model_choice and model_choice.startswith("text-embedding-3-small"):
        provider, model, default_dim = "openai", "text-embedding-3-small", 1536
    elif model_choice and model_choice.startswith("text-embedding-3-large"):
        provider, model, default_dim = "openai", "text-embedding-3-large", 3072
    else:
        provider, model, default_dim = "minilm", "sentence-transformers/all-MiniLM-L6-v2", 384

    dim = IntPrompt.ask("Embedding dimension", default=default_dim)

    cfg = QMemConfig(
        openai_api_key=(openai_key or None),
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_key,
        embed_provider=provider,
        embed_model=model,
        embed_dim=dim,
        default_collection=None,
    )
    cfg.save(cfg_path)
    console.print(f"[green]Saved[/green] config to {cfg_path}")


# -----------------------------
# create
# -----------------------------

@app.command("create", help="Create a Qdrant collection interactively")
def create_collection(
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name"),
    dim: Optional[int] = typer.Option(None, "--dim", help="Vector size"),
    distance: Optional[str] = typer.Option(
        None,
        "--distance",
        "-d",
        help="Distance metric: cosine|dot|euclid (if omitted, you'll be prompted)",
    ),
) -> None:
    cfg = QMemConfig.load(CONFIG_PATH)
    collection = collection or Prompt.ask("collection_name")
    q = QMem(cfg, collection=collection)

    # exists?
    try:
        q.client.get_collection(collection)
        console.print(f"[yellow]Collection already exists:[/yellow] {collection}")
        return
    except Exception:
        pass

    dim = dim or IntPrompt.ask("Embedding vector size for this collection", default=cfg.embed_dim or 1024)

    # If not supplied via flag, ask interactively
    dist_key = distance.strip().lower() if distance else None
    if not dist_key:
        dist_key = qs.select(
            "Distance metric:",
            choices=list(_DISTANCE_MAP.keys()),
            pointer="➤",
            use_shortcuts=False,
            default="cosine",
        ).ask()

    if dist_key not in _DISTANCE_MAP:
        _fail(f"Invalid distance: {dist_key!r}. Choose from: {', '.join(_DISTANCE_MAP)}")

    q.ensure_collection(create_if_missing=True, distance=_DISTANCE_MAP[dist_key], vector_size=dim)
    console.print(f"[green]Collection created:[/green] {collection} (dim={dim}, distance={dist_key})")


# -----------------------------
# ingest
# -----------------------------

@app.command("ingest", help="Ingest JSON/JSONL into a collection (one fixed embed field for all rows; vectors respected if present)")
def ingest(
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name"),
    data: Optional[Path] = typer.Option(None, "--data", "-f", help="Path to JSON/JSONL"),
    embed_field: Optional[str] = typer.Option(
        None,
        "--embed-field",
        "-e",
        help="Field to embed for ALL records (prompted if omitted)",
    ),
    payload: Optional[str] = typer.Option(None, "--payload", "-p", help="Comma-separated payload fields to keep"),
) -> None:
    cfg = QMemConfig.load(CONFIG_PATH)
    collection = collection or Prompt.ask("collection_name")
    q = QMem(cfg, collection=collection)
    _ensure_collection_exists(q, collection)

    data_path = data or Path(Prompt.ask("data file path (JSON or JSONL)"))
    if not data_path:
        _fail("Data file path is required")

    records = _read_records(data_path)
    if not records:
        _fail(f"No records found in {data_path}")

    # Build dynamic list of candidate string fields from the data itself
    candidates = _collect_string_fields(records)
    if not candidates and not any("vector" in r for r in records):
        _fail(
            "No string fields discovered in the data and no precomputed 'vector' found.\n"
            "Add a string field to embed or include a 'vector' array in each record."
        )

    # If user did not pass --embed-field, prompt dynamically
    if not embed_field:
        embed_field = qs.select(
            "Select the field to embed for ALL records:",
            choices=candidates if candidates else ["(no string fields found)"],
            pointer="➤",
            use_shortcuts=False,
        ).ask()

    if not embed_field or (candidates and embed_field not in candidates):
        _fail(f"Invalid embed field: {embed_field!r}. Choose one of: {', '.join(candidates) if candidates else '[]'}")

    payload_keys = _parse_payload_keys(
        payload or Prompt.ask("payload fields (comma-separated, empty = all except embedded)", default="")
    )

    # Construct items; each record uses the SAME embed_field (unless vector is present)
    items: List[IngestItem] = []
    for d in records:
        known = {
            "vector": d.get("vector"),  # if present, will be used directly
            "embed_field": embed_field,
            "graph": d.get("graph"),
            "tags": d.get("tags"),
        }

        # Ensure the selected embed_field is available at top-level
        if embed_field in d:
            known[embed_field] = d[embed_field]

        # Also copy common text fields if present
        for k in ("query", "response", "sql_query", "doc_id"):
            if k in d:
                known[k] = d[k]

        # Extra payload keys (anything not already in known)
        extras = {k: v for k, v in d.items() if k not in known}
        if extras:
            known["extra"] = extras

        items.append(IngestItem(**known))

    # Keep the embedded field text in payload as well
    n = q.ingest(items, payload_keys=payload_keys, include_embed_in_payload=True)
    console.print(
        f"[green]Upserted[/green] {n} items into [bold]{collection}[/bold] "
        f"(embed_field={embed_field}, embedded text [bold]stored[/bold] in payload)."
    )


# -----------------------------
# retrieve (always show query + response)
# -----------------------------

@app.command("retrieve", help="Vector search and show top-k with payload (does not create collections)")
def retrieve(
    query: Optional[str] = typer.Argument(None, help="Search query (if omitted, you'll be prompted)"),
    k: Optional[int] = typer.Option(None, "--k", "-k", help="Top K"),
    json_out: bool = typer.Option(False, "--json", help="Print raw JSON results"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name"),
) -> None:
    cfg = QMemConfig.load(CONFIG_PATH)
    collection = collection or Prompt.ask("collection_name")
    q = QMem(cfg, collection=collection)
    _ensure_collection_exists(q, collection)

    if not query:
        query = Prompt.ask("query:")
    if k is None:
        k = IntPrompt.ask("top_k:", default=5)

    results = q.search(query, top_k=k)

    if json_out:
        console.print_json(data=[r.model_dump() for r in results])
        return

    show_keys = ["query", "response"]  # fixed columns for CLI

    table = Table(title=f"Top {k} results", box=box.ROUNDED)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    for key in show_keys:
        table.add_column(key)

    for i, r in enumerate(results, 1):
        payload = r.payload or {}
        row = [str(i), f"{r.score:.4f}"] + [_truncate(payload.get(k)) for k in show_keys]
        table.add_row(*row)

    console.print(table)
