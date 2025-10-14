from __future__ import annotations

import json
import re
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
from .config import CONFIG_PATH, QMemConfig, FILTERS_DIR
from .schemas import IngestItem

# NEW: import cached API wrappers
from .api import retrieve as api_retrieve, retrieve_by_filter as api_retrieve_by_filter

__all__ = ["app"]

app = typer.Typer(help="qmem — Memory for ingestion & retrieval (Qdrant cloud or Chroma local)")
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
    """
    For Qdrant: verify collection exists (no creation). For Chroma: collections are lazy / created on first use,
    so we skip existence check to avoid creating as a side-effect.
    """
    backend = getattr(q, "_backend", "qdrant")
    if backend == "chroma":
        return  # no-op; Chroma collections are created on first use
    try:
        # Qdrant check
        q.ensure_collection(create_if_missing=False)
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

# default annotation helpers
def _used_default(val, default) -> bool:
    return val == default

def _ann(val, default) -> str:
    return " (default)" if _used_default(val, default) else ""

# save filters for reuse
def _save_filter_to_file(filter_obj: dict, name: Optional[str] = None) -> Path:
    """
    Save filter JSON next to the active config, under <cwd>/.qmem/filters/<name>.json.
    If name is omitted/blank, uses 'latest.json'.
    """
    FILTERS_DIR.mkdir(parents=True, exist_ok=True)
    filename = (name or "latest").strip() or "latest"
    safe = "".join(c for c in filename if c.isalnum() or c in ("-", "_")).rstrip(".")
    path = FILTERS_DIR / f"{safe}.json"
    path.write_text(json.dumps(filter_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

# -----------------------------
# Filter Builder helpers
# -----------------------------

def _ask_condition(group_name: str) -> dict:
    ctype = qs.select(
        f"[{group_name}] Pick condition type:",
        choices=[
            "match (exact)",
            "match-any (array)",
            "range (gte/lte/gt/lt)",
            "is_null",
            "exists",
            "has_id",
            "geo_radius",
            "geo_bounding_box",
        ],
        pointer="➤",
        use_shortcuts=False,
    ).ask()

    if ctype == "match (exact)":
        key = Prompt.ask("Key")
        val = Prompt.ask("Value (exact)")
        return {"key": key, "match": {"value": val}}

    if ctype == "match-any (array)":
        key = Prompt.ask("Key")
        raw = Prompt.ask("Values (comma-separated)")
        vals = [s.strip() for s in raw.split(",") if s.strip()]
        return {"key": key, "match": {"any": vals}}

    if ctype == "range (gte/lte/gt/lt)":
        key = Prompt.ask("Key")
        gte = Prompt.ask("gte (blank=skip)", default="", show_default=False)
        lte = Prompt.ask("lte (blank=skip)", default="", show_default=False)
        gt = Prompt.ask("gt (blank=skip)", default="", show_default=False)
        lt = Prompt.ask("lt (blank=skip)", default="", show_default=False)
        rng: Dict[str, object] = {}
        if gte != "":
            try:
                rng["gte"] = float(gte)
            except Exception:
                rng["gte"] = gte
        if lte != "":
            try:
                rng["lte"] = float(lte)
            except Exception:
                rng["lte"] = lte
        if gt != "":
            try:
                rng["gt"] = float(gt)
            except Exception:
                rng["gt"] = gt
        if lt != "":
            try:
                rng["lt"] = float(lt)
            except Exception:
                rng["lt"] = lt
        return {"key": key, "range": rng}

    if ctype == "is_null":
        key = Prompt.ask("Key")
        return {"is_null": {"key": key}}

    if ctype == "exists":
        key = Prompt.ask("Key")
        return {"has_field": {"key": key}}

    if ctype == "has_id":
        raw = Prompt.ask("IDs (comma-separated)")
        ids = [s.strip() for s in raw.split(",") if s.strip()]
        return {"has_id": {"values": ids}}

    if ctype == "geo_radius":
        key = Prompt.ask("Key (payload field with {lat,lon})")
        lat = float(Prompt.ask("center.lat"))
        lon = float(Prompt.ask("center.lon"))
        radius = float(Prompt.ask("radius (meters)"))
        return {"key": key, "geo_radius": {"center": {"lat": lat, "lon": lon}, "radius": radius}}

    if ctype == "geo_bounding_box":
        key = Prompt.ask("Key (payload field with {lat,lon})")
        tl = Prompt.ask("top_left lat,lon (e.g. 40.73,-74.10)")
        br = Prompt.ask("bottom_right lat,lon (e.g. 40.01,-71.12)")
        tl_lat, tl_lon = [float(x.strip()) for x in tl.split(",")]
        br_lat, br_lon = [float(x.strip()) for x in br.split(",")]
        return {
            "key": key,
            "geo_bounding_box": {
                "top_left": {"lat": tl_lat, "lon": tl_lon},
                "bottom_right": {"lat": br_lat, "lon": br_lon},
            },
        }

    return {}

def _build_filter_interactive() -> dict:
    filter_obj: Dict[str, list] = {}

    groups = qs.checkbox(
        "Choose groups to include (space to toggle):",
        choices=[
            qs.Choice("must", checked=True),
            qs.Choice("should", checked=False),
            qs.Choice("must_not", checked=False),
        ],
    ).ask()

    def ask_group(name: str) -> None:
        n = IntPrompt.ask(f"How many conditions in {name.upper()}?", default=0)
        if n <= 0:
            return
        conds = []
        for _ in range(1, n + 1):
            conds.append(_ask_condition(name.upper()))
        filter_obj[name] = conds

    if "must" in groups:
        ask_group("must")
    if "must_not" in groups:
        ask_group("must_not")
    if "should" in groups:
        ask_group("should")
        if "should" in filter_obj and filter_obj["should"]:
            ms_default = 1
            ms = IntPrompt.ask("Minimum number of SHOULD to match", default=ms_default)
            filter_obj["min_should"] = ms
            console.print(f"[dim]min_should = {ms}{_ann(ms, ms_default)}[/dim]")

    console.print("Preview filter →", style="bold")
    console.print_json(data=filter_obj)
    yn = Prompt.ask("Run with this filter? (y/N)", default="y")
    if (yn or "").lower().startswith("y"):
        return filter_obj
    return _build_filter_interactive()  # restart wizard to re-edit

# -----------------------------
# init
# -----------------------------

@app.command("init", help="Configure backend (Qdrant/Chroma), keys + embedding model (saved to ./.qmem/config.toml)")
def init_cmd() -> None:
    cfg_path = CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("qmem init — backend, keys, and embedding provider/model", style="bold")

    # 0) Vector store backend selection
    vector_store = qs.select(
        "Choose vector store backend:",
        choices=[
            qs.Choice("qdrant", checked=True),
            qs.Choice("chroma", checked=False),
        ],
        pointer="➤",
        use_shortcuts=False,
    ).ask()

    # Qdrant (cloud) or Chroma (local path)
    qdrant_url = ""
    qdrant_key = ""
    chroma_path = ""
    if vector_store == "qdrant":
        qdrant_url = Prompt.ask("Qdrant URL (e.g., https://xxxx.cloud.qdrant.io)")
        qdrant_key = Prompt.ask("Qdrant API key", password=True)
    else:
        chroma_path = Prompt.ask(
            "Local Chroma path (blank = ./.qmem/chroma)", default="", show_default=False
        ) or str((CONFIG_PATH.parent / "chroma").resolve())

    # NEW) Cache backend selection
    cache_backend = qs.select(
        "Cache backend:",
        choices=[
            qs.Choice("local", checked=True),
            qs.Choice("redis", checked=False),
        ],
        pointer="➤",
        use_shortcuts=False,
    ).ask()

    # 1) Embedding provider
    provider = qs.select(
        "Embedding provider:",
        choices=["openai", "gemini", "voyage", "minilm"],
        pointer="➤",
        use_shortcuts=False,
    ).ask()

    # 2) Provider-specific models + keys
    openai_key = ""
    hf_key = ""
    gemini_key = ""
    voyage_key = ""
    default_dim = 1024  # will be overridden per-provider
    dim: Optional[int] = None  # if set inside a branch, we won't re-prompt later

    if provider == "openai":
        model = qs.select(
            "OpenAI embedding model:",
            choices=[
                "text-embedding-3-small (1536)",
                "text-embedding-3-large (3072)",
            ],
            pointer="➤",
            use_shortcuts=False,
        ).ask()
        if model.startswith("text-embedding-3-small"):
            model, default_dim = "text-embedding-3-small", 1536
        else:
            model, default_dim = "text-embedding-3-large", 3072
        openai_key = Prompt.ask("OpenAI API key", password=True)

    elif provider == "gemini":
        model = qs.select(
            "Gemini (Google) embedding model:",
            choices=[
                "models/embedding-001 (768)",
            ],
            pointer="➤",
            use_shortcuts=False,
        ).ask()
        model, default_dim = "models/embedding-001", 768
        gemini_key = Prompt.ask("Gemini API key", password=True)

    elif provider == "voyage":
        model = qs.select(
            "Voyage embedding model:",
            choices=[
                "voyage-3-large",
                "voyage-3.5",
                "voyage-3.5-lite",
                "voyage-code-3",
                "voyage-finance-2",
                "voyage-law-2",
                "voyage-code-2",
            ],
            pointer="➤",
            use_shortcuts=False,
        ).ask()

        # Default dims by model + conditional override prompt
        if model in {"voyage-3-large", "voyage-3.5", "voyage-3.5-lite", "voyage-code-3"}:
            default_dim = 1024
            dim = IntPrompt.ask("Embedding dimension (256, 512, 1024, or 2048)", default=default_dim)
        elif model == "voyage-finance-2":
            default_dim = 1024
            dim = 1024
        elif model == "voyage-law-2":
            default_dim = 1024
            dim = 1024
        elif model == "voyage-code-2":
            default_dim = 1536
            dim = 1536
        else:
            default_dim = 1024  # fallback

        voyage_key = Prompt.ask("Voyage API key", password=True)

    else:  # minilm (Hugging Face Inference API)
        model, default_dim = "sentence-transformers/all-MiniLM-L6-v2", 384
        hf_key = Prompt.ask("Hugging Face API key (Inference API)", password=True)

    # 3) Dimension: only ask here if not already fixed/asked above
    if dim is None:
        dim = IntPrompt.ask("Embedding dimension", default=default_dim)

    # 4) Save config (now includes cache_backend)
    cfg = QMemConfig(
        vector_store=vector_store,
        chroma_path=(chroma_path or None),
        cache_backend=cache_backend,  # <-- NEW
        qdrant_url=qdrant_url or "http://localhost:6333",
        qdrant_api_key=qdrant_key or "",
        openai_api_key=(openai_key or None),
        hf_api_key=(hf_key or None),
        gemini_api_key=(gemini_key or None),
        voyage_api_key=(voyage_key or None),
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

@app.command("create", help="Create a collection interactively")
def create_collection(
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name"),
    dim: Optional[int] = typer.Option(None, "--dim", help="Vector size"),
    distance: Optional[str] = typer.Option(
        None,
        "--distance",
        "-d",
        help="Distance metric: cosine|dot|euclid (Qdrant only; ignored for Chroma)",
    ),
) -> None:
    cfg = QMemConfig.load(CONFIG_PATH)
    collection = collection or Prompt.ask("collection_name")
    q = QMem(cfg, collection=collection)

    backend = getattr(q, "_backend", "qdrant")
    if backend == "qdrant":
        # Try to see if exists
        try:
            # Will raise if not exists and create_if_missing=False
            q.ensure_collection(create_if_missing=False)
            console.print(f"[yellow]Collection already exists:[/yellow] {collection}")
            return
        except Exception:
            pass

    dim = dim or IntPrompt.ask("Embedding vector size for this collection", default=cfg.embed_dim or 1024)

    # Persist chosen dim to config for consistency
    if cfg.embed_dim != dim:
        cfg.embed_dim = dim
        cfg.save(CONFIG_PATH)

    # Distance selection (Qdrant only)
    dist_key = distance.strip().lower() if distance else None
    if backend == "qdrant":
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
    else:
        # Chroma ignores distance & vector size; we still record dim in config for the embedder
        q.ensure_collection(create_if_missing=True)
        console.print(f"[green]Collection ready (Chroma):[/green] {collection} (dim={dim}, distance=ignored)")

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

    candidates = _collect_string_fields(records)
    if not candidates and not any("vector" in r for r in records):
        _fail(
            "No string fields discovered in the data and no precomputed 'vector' found.\n"
            "Add a string field to embed or include a 'vector' array in each record."
        )

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
        payload or Prompt.ask("payload fields (comma-separated, empty = keep all fields)", default="")
    )

    items: List[IngestItem] = []
    for d in records:
        known = dict(d)
        known["embed_field"] = embed_field
        if embed_field in d:
            known[embed_field] = d[embed_field]
        for k in ("query", "response", "sql_query", "doc_id", "graph", "tags"):
            if k in d:
                known[k] = d[k]
        known.pop("extra", None)
        items.append(IngestItem(**known))

    n = q.ingest(items, payload_keys=payload_keys, include_embed_in_payload=True)
    console.print(
        f"[green]Upserted[/green] {n} items into [bold]{collection}[/bold] "
        f"(embed_field={embed_field}, embedded text [bold]stored[/bold] in payload)."
    )

# -----------------------------
# retrieve (dynamic columns via prompt)
# -----------------------------

@app.command("retrieve", help="Vector search and show top-k with chosen payload columns")
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
        query = Prompt.ask("query")
    if k is None:
        k_default = 5
        k = IntPrompt.ask("top_k:", default=k_default)
        console.print(f"[dim]Run options → top_k = {k}{_ann(k, k_default)}[/dim]")

    # NEW: use cached API
    try:
        results = api_retrieve(collection=collection, query=query, k=k, cfg=cfg)
    except Exception as e:
        console.print(f"[red]Request failed:[/red] {e}")
        raise typer.Exit(code=2)

    if json_out:
        console.print_json(data=[r.model_dump() for r in results])
        return

    raw_cols = Prompt.ask(
        "Columns to display (comma-separated; empty = ALL payload fields)",
        default="",
        show_default=False,
    )
    if raw_cols.strip():
        show_keys = [s.strip() for s in raw_cols.split(",") if s.strip()]
    else:
        freq: Counter[str] = Counter()
        for r in results:
            for key, val in (r.payload or {}).items():
                if isinstance(val, (str, int, float, bool)) and (str(val).strip() if isinstance(val, str) else True):
                    freq[key] += 1
        show_keys = [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]

    table = Table(title=f"Top {k} results", box=box.ROUNDED)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    for key in show_keys:
        table.add_column(key)

    for i, r in enumerate(results, 1):
        payload = r.payload or {}
        row = [str(i), f"{r.score:.4f}"] + [
            str(payload.get(col, "")).replace("\n", " ") for col in show_keys
        ]
        table.add_row(*row)

    console.print(table)

# -----------------------------
# Friendly Qdrant 400 parsing
# -----------------------------

_index_err_re = re.compile(
    # Allow quoted or unquoted field names; simpler escaping
    r'Index required but not found for\s+["\']?(?P<field>[^"\']+)["\']?\s+of one of the following types:\s+\[(?P<types>[^\]]+)\]',
    re.IGNORECASE,
)

def _maybe_hint_create_index(error_msg: str) -> None:
    m = _index_err_re.search(error_msg or "")
    if not m:
        return
    field = m.group("field").strip()
    types = [t.strip().lower() for t in m.group("types").split(",")]
    console.print(
        "\n[yellow]Hint[/yellow]: Qdrant needs a payload index to filter on this field.\n"
        f"Run: [bold]qmem index[/bold] and add field [bold]{field}[/bold] with type "
        f"[bold]{' / '.join(types)}[/bold]."
    )

# -----------------------------
# filter (payload-only) — prompts for columns
# -----------------------------

@app.command("filter", help="Filter payload (no vector search) — interactive builder")
def filter_cmd() -> None:
    cfg = QMemConfig.load(CONFIG_PATH)
    collection = Prompt.ask("collection_name")
    q = QMem(cfg, collection=collection)
    _ensure_collection_exists(q, collection)

    filter_json = _build_filter_interactive()
    limit_default = 100
    limit = IntPrompt.ask("Limit", default=limit_default)
    console.print(f"[dim]Run options → limit = {limit}{_ann(limit, limit_default)}[/dim]")

    yn_save = Prompt.ask("Save this filter for reuse? (y/N)", default="y")
    if (yn_save or "").lower().startswith("y"):
        default_name = "latest"
        name = Prompt.ask("Filter name (stored under .qmem/filters/)", default=default_name)
        saved_path = _save_filter_to_file(filter_json, name)
        console.print(f"[green]Saved filter[/green] → {saved_path}")

    try:
        # FIX: actually apply the built payload filter
        results, _ = q.scroll_filter(query_filter=filter_json, limit=limit)
    except Exception as e:
        msg = str(e)
        console.print(f"[red]Request failed:[/red] {msg}")
        _maybe_hint_create_index(msg)
        raise typer.Exit(code=2)

    raw_cols = Prompt.ask(
        "Columns to display (comma-separated; empty = ALL payload fields)",
        default="",
        show_default=False,
    )
    if raw_cols.strip():
        show_keys = [s.strip() for s in raw_cols.split(",") if s.strip()]
    else:
        freq: Counter[str] = Counter()
        for r in results:
            for key, val in (r.payload or {}).items():
                if isinstance(val, (str, int, float, bool)) and (str(val).strip() if isinstance(val, str) else True):
                    freq[key] += 1
        show_keys = [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]

    # Rich table rendering
    table = Table(title=f"Filtered results (limit {limit})", box=box.ROUNDED)
    table.add_column("#", justify="right")
    for key in show_keys:
        table.add_column(key)

    for i, r in enumerate(results, 1):
        payload = r.payload or {}
        row = [str(i)] + [str(payload.get(col, "")).replace("\n", " ") for col in show_keys]
        table.add_row(*row)

    console.print(table)

# -----------------------------
# retrieve_filter (hybrid) — prompts for columns
# -----------------------------

@app.command("retrieve_filter", help="Vector search scoped by a payload filter — interactive builder")
def retrieve_filter_cmd() -> None:
    cfg = QMemConfig.load(CONFIG_PATH)
    collection = Prompt.ask("collection_name")
    q = QMem(cfg, collection=collection)
    _ensure_collection_exists(q, collection)

    query = Prompt.ask("query")
    topk_default = 5
    topk = IntPrompt.ask("top_k:", default=topk_default)
    console.print(f"[dim]Run options → top_k = {topk}{_ann(topk, topk_default)}[/dim]")

    filter_json = _build_filter_interactive()

    yn_save = Prompt.ask("Save this filter for reuse? (y/N)", default="y")
    if (yn_save or "").lower().startswith("y"):
        default_name = "latest"
        name = Prompt.ask("Filter name (stored under .qmem/filters/)", default=default_name)
        saved_path = _save_filter_to_file(filter_json, name)
        console.print(f"[green]Saved filter[/green] → {saved_path}")

    # NEW: use cached API for hybrid search
    try:
        results = api_retrieve_by_filter(
            collection=collection,
            filter=filter_json,
            k=topk,
            query=query,
            cfg=cfg,
        )
    except Exception as e:
        msg = str(e)
        console.print(f"[red]Request failed:[/red] {msg}")
        _maybe_hint_create_index(msg)
        raise typer.Exit(code=2)

    raw_cols = Prompt.ask(
        "Columns to display (comma-separated; empty = ALL payload fields)",
        default="",
        show_default=False,
    )
    if raw_cols.strip():
        show_keys = [s.strip() for s in raw_cols.split(",") if s.strip()]
    else:
        freq: Counter[str] = Counter()
        for r in results:
            for key, val in (r.payload or {}).items():
                if isinstance(val, (str, int, float, bool)) and (str(val).strip() if isinstance(val, str) else True):
                    freq[key] += 1
        show_keys = [k for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]

    # Rich table rendering
    table = Table(title=f"Top {topk} results (filtered)", box=box.ROUNDED)
    table.add_column("#", justify="right")
    for key in show_keys:
        table.add_column(key)

    for i, r in enumerate(results, 1):
        payload = r.payload or {}
        row = [str(i)] + [str(payload.get(col, "")).replace("\n", " ") for col in show_keys]
        table.add_row(*row)

    console.print(table)

# -----------------------------
# mongo (mirror an existing collection → MongoDB)
# -----------------------------

@app.command("mongo", help="Mirror an existing collection's payloads into MongoDB (Qdrant or Chroma backend)")
def mongo_cmd() -> None:
    cfg = QMemConfig.load(CONFIG_PATH)
    collection = Prompt.ask("collection_name")
    q = QMem(cfg, collection=collection)

    # Allow both backends now (Qdrant and Chroma)
    _ensure_collection_exists(q, collection)

    console.print("[bold]Discovering payload fields (sampling up to 200 docs)...[/bold]")
    # sample to collect keys
    try:
        sample_results, _ = q.scroll_filter(query_filter={}, limit=200)
    except Exception as e:
        console.print(f"[red]Failed to sample collection:[/red] {e}")
        raise typer.Exit(code=2)

    # Build a frequency map of simple scalar keys
    freq: Counter[str] = Counter()
    for r in sample_results:
        for k, v in (r.payload or {}).items():
            if isinstance(v, (str, int, float, bool)) and (str(v).strip() if isinstance(v, str) else True):
                freq[k] += 1

    # Present dynamic field list (checkbox). Empty selection => FULL payload.
    choices = [qs.Choice(k, checked=False) for k, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]
    if not choices:
        console.print("[yellow]No scalar payload fields found in sample; defaulting to FULL payload.[/yellow]")
        selected_fields = []
    else:
        selected_fields = qs.checkbox(
            "Select fields to store in Mongo (space to toggle; ENTER to confirm; empty = FULL payload)",
            choices=choices,
        ).ask() or []

    mongo_uri = Prompt.ask(
        "Mongo URI (blank = mongodb://127.0.0.1:27017)", default="", show_default=False
    ) or "mongodb://127.0.0.1:27017"
    mongo_db = Prompt.ask(
        "Mongo database (blank = qmem)", default="", show_default=False
    ) or "qmem"
    mongo_coll = Prompt.ask(
        f"Mongo collection name (blank = {collection})", default="", show_default=False
    ) or collection

    confirm = Prompt.ask(
        "Mirror ALL points from the selected backend → Mongo? (y/N)", default="y"
    )
    if not (confirm or "").lower().startswith("y"):
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(code=0)

    # None => FULL payload; else restrict to selected set
    mongo_keys = set(selected_fields) if selected_fields else None

    total = q.mirror_to_mongo(
        mongo_uri=mongo_uri,
        mongo_db=mongo_db,
        mongo_coll=mongo_coll,
        mongo_keys=mongo_keys,
        batch_size=1000,  # backend-specific page size
    )
    console.print(
        f"[green]Mirrored[/green] {total} documents to Mongo (db={mongo_db}, coll={mongo_coll})."
    )

# =============================
# Letta integration + Chatbot
# =============================

# We import here to avoid hard-dependency issues during base CLI operations
import os as _os
import sys as _sys
import shutil as _shutil
import subprocess as _subprocess
import webbrowser as _webbrowser
import time as _time
from rich import print as _rprint
import os

# Letta client import (SDK)
try:
    from letta_client import Letta as _Letta
    from letta_client.errors import UnprocessableEntityError as _UnprocessableEntityError
    _HAVE_LETTA = True
except Exception:
    _HAVE_LETTA = False

# QMem-backed Letta tools
try:
    from qmem.letta_tools import (
        QMemInsertTool as _QMemInsertTool,
        QMemSearchTool as _QMemSearchTool,
    )
except Exception:
    _QMemInsertTool = None
    _QMemSearchTool = None

def _require_letta():
    if not _HAVE_LETTA:
        _rprint("[red]letta-client is unavailable. Reinstall qmem or ensure dependencies are installed.[/red]")
        raise typer.Exit(code=2)

def _ensure_env_default(key: str, default: str) -> str:
    if not _os.environ.get(key):
        _os.environ[key] = default
    return _os.environ[key]

def _warn_if_not_qdrant():
    cfg = QMemConfig.load(CONFIG_PATH)
    backend = (getattr(cfg, "vector_store", "qdrant") or "qdrant").lower()
    if backend != "qdrant":
        _rprint(f"[yellow]Warning:[/yellow] QMem backend is '{backend}'. Archival writes will NOT go to Qdrant. "
                "Run `qmem init` here or export QMEM_VECTOR_STORE=qdrant, QMEM_QDRANT_URL, QMEM_QDRANT_API_KEY.")

def _upsert_qmem_archival_agent(client: "_Letta") -> str:
    """Create (or find) an agent that mirrors `letta_demo` tool config and return its id."""
    name = "qmem-archival-agent"
    system = (
        "When the user says 'remember ...' with no qualifier, interpret it as a request "
        "for long-term archival memory. ALWAYS and EXCLUSIVELY call the tool `archival_memory_insert` "
        "(QMem) to persist it. Use only temporary/session memory if the user explicitly says "
        "'temporary' or 'scratchpad'. When the user states profile facts (e.g., 'my name is X'), call `archival_memory_insert` with content exactly like 'my name is X'."
    )
    tools = [
        "send_message",
        "conversation_search",
        "archival_memory_insert",
        "archival_memory_search",
    ]

    # Try to find an existing agent with same name
    try:
        existing = client.agents.list()
        for a in existing or []:
            if getattr(a, "name", "") == name:
                return getattr(a, "id", None) or a.get("id")  # type: ignore[attr-defined]
    except Exception:
        pass

    # Create new if not found
    agent = client.agents.create(
        name=name,
        model=os.environ.get("LETTA_MODEL", "letta/letta-free"),
        embedding=os.environ.get("LETTA_EMBEDDING", "letta/letta-free"),
        include_base_tools=False,  # CHANGED: do not include base tools to avoid generic memory_insert
        tools=tools,
        system=system,
        memory_blocks=[{"label":"persona","value":"I use QMem for long-term memory."}],
        enable_sleeptime=False,
    )
    return getattr(agent, "id", None) or agent.get("id")  # type: ignore[attr-defined]

@app.command("letta-register-tools")
def letta_register_tools(base_url: Optional[str] = None, token: Optional[str] = None):
    """Register QMem archival tools with a running Letta server (no alias tools)."""
    _require_letta()
    if _QMemInsertTool is None or _QMemSearchTool is None:
        console.print("[red]Could not import qmem.letta_tools. Ensure src/qmem/letta_tools.py exists.[/red]")
        raise typer.Exit(code=2)

    base_url = base_url or _ensure_env_default("LETTA_BASE_URL", "http://localhost:8283")
    token = token or _os.environ.get("LETTA_TOKEN")

    # Construct client (Bearer token if provided)
    client = _Letta(base_url=base_url, token=token) if token else _Letta(base_url=base_url)

    # Register tools (idempotent)
    try:
        client.tools.add(tool=_QMemInsertTool())
        client.tools.add(tool=_QMemSearchTool())
    except Exception as e:
        _rprint(f"[red]Failed to register tools:[/red] {e}")
        _rprint("• Is the Letta server running at", base_url, "?")
        _rprint("• If secured, export LETTA_TOKEN before running this command.")
        raise typer.Exit(code=2)

    _rprint(f"[bold green]Registered[/] QMem archival tools at {base_url}")

@app.command("letta-demo")
def letta_demo():
    """Create an agent + quick smoke test (requires local Letta server)."""
    _require_letta()
    base_url = _ensure_env_default("LETTA_BASE_URL", "http://localhost:8283")
    token = _os.environ.get("LETTA_TOKEN")
    client = _Letta(base_url=base_url, token=token) if token else _Letta(base_url=base_url)

    # idempotent tool registration
    client.tools.add(tool=_QMemInsertTool())
    client.tools.add(tool=_QMemSearchTool())

    # NOTE: Newer Letta builds require using the 'letta' provider handle.
    try:
        agent = client.agents.create(
            model=os.environ.get("LETTA_MODEL", "letta/letta-free"),
            embedding=os.environ.get("LETTA_EMBEDDING", "letta/letta-free"),
            include_base_tools=False,  # CHANGED
            tools=[
                "send_message",
                "conversation_search",
                "archival_memory_insert",
                "archival_memory_search",
            ],
            system=(
                "When the user says 'remember ...' with no qualifier, interpret it as a request "
                "for long-term archival memory. ALWAYS and EXCLUSIVELY call the tool `archival_memory_insert` "
                "(QMem) to persist it. Use only temporary/session memory if the user explicitly says "
                "'temporary' or 'scratchpad'. When the user states profile facts (e.g., 'my name is X'), call `archival_memory_insert` with content exactly like 'my name is X'."
            ),
            memory_blocks=[{"label":"persona","value":"I use QMem for long-term memory."}],
            enable_sleeptime=False,
        )
    except _UnprocessableEntityError as e:
        _rprint("[yellow]Agent create validation failed, retrying with base tools only...[/yellow]", e)
        agent = client.agents.create(
            model=os.environ.get("LETTA_MODEL", "letta/letta-free"),
            embedding=os.environ.get("LETTA_EMBEDDING", "letta/letta-free"),
            include_base_tools=False,  # CHANGED
            tools=[
                "send_message",
                "conversation_search",
                "archival_memory_insert",
                "archival_memory_search",
            ],
            system=(
                "When the user says 'remember ...' with no qualifier, interpret it as a request "
                "for long-term archival memory. ALWAYS and EXCLUSIVELY call the tool `archival_memory_insert` "
                "(QMem) to persist it. Use only temporary/session memory if the user explicitly says "
                "'temporary' or 'scratchpad'. When the user states profile facts (e.g., 'my name is X'), call `archival_memory_insert` with content exactly like 'my name is X'."
            ),
            memory_blocks=[{"label":"persona","value":"I use QMem for long-term memory."}],
            enable_sleeptime=False,
        )
    _rprint("Agent:", getattr(agent, "id", "(unknown)"))

    # Quick local test of the tool functions
    _QMemInsertTool().run(content="The default build target is release mode.")
    out = _QMemSearchTool().run(query="What is the build target?", k=3)
    _rprint("[bold cyan]QMem search:[/]", out or "(no hits)")

@app.command("chatbot")
def chatbot():
    """
    Start the Letta Chatbot Template and force it to use the same QMem archival tools as `letta-demo`:
      - Registers QMem archival tools on the running Letta server
      - Creates (or reuses) a server-side agent with those tools
      - Exposes NEXT_PUBLIC_DEFAULT_AGENT_ID so the UI attaches to that agent (no agent creation from UI)
    """
    # Require Node & npm & git
    for exe in ("node", "npm", "git"):
        if _shutil.which(exe) is None:
            console.print(f"[red]{exe} not found. Please install Node.js (>=18), npm, and git.[/red]")
            raise typer.Exit(code=2)

    _warn_if_not_qdrant()

    base_url = _ensure_env_default("LETTA_BASE_URL", "http://localhost:8283")

    # --- Ensure tools are registered BEFORE the UI starts, and create a server agent ---
    if _HAVE_LETTA and _QMemInsertTool is not None and _QMemSearchTool is not None:
        token = _os.environ.get("LETTA_TOKEN")
        client = _Letta(base_url=base_url, token=token) if token else _Letta(base_url=base_url)
        try:
            client.tools.add(tool=_QMemInsertTool())
            client.tools.add(tool=_QMemSearchTool())
            _rprint("[green]QMem archival tools registered with Letta server.[/green]")
        except Exception as e:
            _rprint("[yellow]Warning:[/yellow] Could not register archival tools automatically:", e)
            _rprint("You can run `qmem letta-register-tools` in another terminal.")
            raise typer.Exit(code=2)

        # Upsert the qmem-archival-agent on the server (mirrors letta_demo)
        try:
            agent_id = _upsert_qmem_archival_agent(client)
            _rprint(f"[green]Using server agent:[/green] {agent_id}")
        except Exception as e:
            _rprint("[red]Failed to create/find server agent:[/red]", e)
            raise typer.Exit(code=2)
    else:
        _rprint("[red]letta-client or qmem.letta_tools missing; cannot register tools or create agent.[/red]")
        raise typer.Exit(code=2)

    # IMPORTANT: Keep assets beside the **current** ./.qmem (not in home)
    project_root_dotqmem = CONFIG_PATH.parent  # typically <cwd>/.qmem
    app_dir = project_root_dotqmem / "letta-chatbot"
    repo = "https://github.com/letta-ai/letta-chatbot-template.git"
    app_dir.parent.mkdir(parents=True, exist_ok=True)

    # Clone or pull
    if not app_dir.exists():
        console.print("[bold]Cloning Letta Chatbot Template...[/bold]")
        _subprocess.run(["git", "clone", repo, str(app_dir)], check=True)
    else:
        console.print("[bold]Updating Letta Chatbot Template...[/bold]")
        try:
            _subprocess.run(["git", "-C", str(app_dir), "pull", "--rebase"], check=True)
        except Exception:
            console.print("[yellow]Git pull failed (local changes?). Continuing with existing tree.[/yellow]")

    # Write .env (pin to our server agent; prevent UI from creating different agents)
    env_lines = [
        f"LETTA_BASE_URL={base_url}",
        f"NEXT_PUBLIC_DEFAULT_AGENT_ID={agent_id}",
        "NEXT_PUBLIC_CREATE_AGENTS_FROM_UI=false",
        "USE_COOKIE_BASED_AUTHENTICATION=false",
    ]
    token = _os.environ.get("LETTA_TOKEN")
    if token:
        env_lines.append(f"LETTA_TOKEN={token}")
    (app_dir / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    # Also keep a default-agent.json around (not used when DEFAULT_AGENT_ID is set), matching letta_demo
    agent_json = {
        "model": "" + os.environ.get("LETTA_MODEL", "letta/letta-free") + "",
        "embedding": "" + os.environ.get("LETTA_EMBEDDING", "letta/letta-free") + "",
        "include_base_tools": False,  # CHANGED
        "tools": [
            "send_message",
            "conversation_search",
            "archival_memory_insert",
            "archival_memory_search",
        ],
        "system": (
            "When the user says 'remember ...' with no qualifier, interpret it as a request "
            "for long-term archival memory. ALWAYS and EXCLUSIVELY call the tool `archival_memory_insert` "
            "(QMem) to persist it. Use only temporary/session memory if the user explicitly says "
            "'temporary' or 'scratchpad'. When the user states profile facts (e.g., 'my name is X'), call `archival_memory_insert` with content exactly like 'my name is X'."
        ),
        "memory_blocks": [
            {"label": "persona", "value": "I am a helpful chatbot with QMem long-term memory."}
        ],
        "enable_sleeptime": False
    }
    (app_dir / "default-agent.json").write_text(json.dumps(agent_json, indent=2), encoding="utf-8")

    # npm install (first run) with retry fallback
    if not (app_dir / "node_modules").exists():
        console.print("[bold]Installing npm deps (first run)...[/bold]")
        try:
            _subprocess.run(["npm", "install"], cwd=str(app_dir), check=True)
        except Exception:
            console.print("[yellow]npm install failed; retrying without optional deps & scripts…[/yellow]")
            env = dict(_os.environ)
            env["CYPRESS_INSTALL_BINARY"] = "0"
            _subprocess.run(
                ["npm", "install", "--no-optional", "--ignore-scripts"],
                cwd=str(app_dir),
                check=True,
                env=env,
            )

    # Start dev server and open browser
    console.print("[bold green]Starting Letta Chatbot on http://localhost:3000[/bold green]")

    def _open():
        _time.sleep(1.5)
        try:
            _webbrowser.open("http://localhost:3000")
        except Exception:
            pass

    try:
        if hasattr(_os, "fork"):
            if _os.fork() == 0:
                _open()
                _os._exit(0)
        else:
            _open()
    except Exception:
        _open()

    _subprocess.run(["npm", "run", "dev"], cwd=str(app_dir), check=False)