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
from . import format_results_table  # shared table formatter

__all__ = ["app"]

app = typer.Typer(help="qmem — Memory for ingestion & retrieval")
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


# default annotation helpers
def _used_default(val, default) -> bool:
    return val == default


def _ann(val, default) -> str:
    return " (default)" if _used_default(val, default) else ""


# save filters for reuse
def _save_filter_to_file(filter_obj: dict, name: Optional[str] = None) -> Path:
    """
    Save filter JSON to ./.qmem/filters/<name>.json.
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

@app.command("init", help="Configure keys + embedding model (saved to ./.qmem/config.toml)")
def init_cmd() -> None:
    cfg_path = CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    console.print("qmem init — set keys and choose embedding model", style="bold")
    openai_key = Prompt.ask(
        "OpenAI API key (leave empty if using MiniLM on Hugging Face)",
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
            "MiniLM (Hugging Face API, 384)",
        ],
        pointer="➤",
        use_shortcuts=False,
    ).ask()

    hf_key = ""
    if model_choice and model_choice.startswith("text-embedding-3-small"):
        provider, model, default_dim = "openai", "text-embedding-3-small", 1536
    elif model_choice and model_choice.startswith("text-embedding-3-large"):
        provider, model, default_dim = "openai", "text-embedding-3-large", 3072
    else:
        provider, model, default_dim = "minilm", "sentence-transformers/all-MiniLM-L6-v2", 384
        hf_key = Prompt.ask("Hugging Face API key (required for MiniLM via HF)", password=True)

    dim = IntPrompt.ask("Embedding dimension", default=default_dim)

    cfg = QMemConfig(
        openai_api_key=(openai_key or None),
        hf_api_key=(hf_key or None),
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

    # NEW: persist chosen dim to config, to keep CLI consistent with qm.create(...)
    if cfg.embed_dim != dim:
        cfg.embed_dim = dim
        cfg.save(CONFIG_PATH)

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

    results = q.search(query, top_k=k)

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
        row = [str(i), f"{r.score:.4f}"] + [_truncate(payload.get(k)) for k in show_keys]
        table.add_row(*row)

    console.print(table)


# -----------------------------
# Friendly Qdrant 400 parsing
# -----------------------------

_INDEX_ERR_RE = re.compile(
    r'Index required but not found for\s+\\"(?P<field>[^"]+)\\"\s+of one of the following types:\s+\[(?P<types>[^\]]+)\]',
    re.IGNORECASE,
)

def _maybe_hint_create_index(error_msg: str) -> None:
    m = _INDEX_ERR_RE.search(error_msg or "")
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

    console.print(format_results_table(results, show_score=False, show_keys=show_keys))


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

    try:
        results = q.search_filtered(query, top_k=topk, query_filter=filter_json)
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

    console.print(format_results_table(results, show_score=False, show_keys=show_keys))


# -----------------------------
# index (interactive payload index creation)
# -----------------------------

@app.command("index", help="Create payload indexes (interactive)")
def index_cmd() -> None:
    """
    Interactively create Qdrant payload indexes for fields you plan to filter on.
    Types supported here: keyword | integer | float | bool
    """
    cfg = QMemConfig.load(CONFIG_PATH)
    collection = Prompt.ask("collection_name")
    q = QMem(cfg, collection=collection)
    _ensure_collection_exists(q, collection)

    console.print("Add fields to index. Leave name empty to finish.", style="bold")

    schema: Dict[str, qmodels.PayloadSchemaType] = {}
    type_map = {
        "keyword": qmodels.PayloadSchemaType.KEYWORD,
        "integer": qmodels.PayloadSchemaType.INTEGER,
        "float": qmodels.PayloadSchemaType.FLOAT,
        "bool": qmodels.PayloadSchemaType.BOOL,
    }

    while True:
        fname = Prompt.ask("Field name (blank to finish)", default="", show_default=False).strip()
        if not fname:
            break
        itype = qs.select(
            f"Index type for '{fname}':",
            choices=["keyword", "integer", "float", "bool"],
            pointer="➤",
            use_shortcuts=False,
        ).ask()
        schema[fname] = type_map[itype]

    if not schema:
        _fail("No fields provided.")

    created = []
    for field, ftype in schema.items():
        try:
            q.client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=ftype,
                wait=True,
            )
            created.append(field)
        except Exception as e:
            console.print(f"[yellow]Skipped[/yellow] {field}: {e}")

    if created:
        console.print(f"[green]Created/verified indexes[/green] on {collection}: {', '.join(created)}")
    else:
        console.print(f"[yellow]No new indexes created[/yellow] on {collection}.")


# -----------------------------
# mongo (mirror an existing Qdrant collection → MongoDB)
# -----------------------------

@app.command("mongo", help="Mirror an existing Qdrant collection's payloads into MongoDB")
def mongo_cmd() -> None:
    cfg = QMemConfig.load(CONFIG_PATH)
    collection = Prompt.ask("collection_name")
    q = QMem(cfg, collection=collection)
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
        "Mirror ALL points from Qdrant → Mongo? (y/N)", default="y"
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
        batch_size=1000,  # scroll batch size
    )
    console.print(
        f"[green]Mirrored[/green] {total} documents to Mongo (db={mongo_db}, coll={mongo_coll})."
    )
