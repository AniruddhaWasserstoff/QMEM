from __future__ import annotations

from typing import Iterable, List, Dict

try:
    from pymongo import MongoClient
    from pymongo.errors import BulkWriteError
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pymongo is required for Mongo mirroring. Install it with:\n"
        "    pip install pymongo>=4.7\n"
        f"Original import error: {e}"
    )


class MongoSink:
    """
    Minimal MongoDB sink for QMem mirroring.

    Usage:
        sink = MongoSink("mongodb://127.0.0.1:27017", "qmem")
        sink.insert_many("my_collection", [{"_id": "uuid", "title": "..."}, ...])
    """

    def __init__(self, uri: str, db: str, *, appname: str = "qmem") -> None:
        if not uri:
            raise ValueError("Mongo URI must not be empty")
        if not db:
            raise ValueError("Mongo DB name must not be empty")

        # Keep a simple, short-lived client for CLI runs.
        self._client = MongoClient(uri, appname=appname)
        self._db = self._client[db]

    # --------------- Public API ---------------

    def insert_many(self, coll: str, docs: Iterable[Dict]) -> int:
        """
        Insert many documents into `coll`. Documents should already contain `_id`
        (we reuse Qdrant IDs for 1:1 mapping). Duplicate _id errors are ignored.

        Returns: number of documents successfully inserted (best-effort).
        """
        docs_list: List[Dict] = list(docs or [])
        if not docs_list:
            return 0

        try:
            res = self._db[coll].insert_many(docs_list, ordered=False)
            return len(res.inserted_ids or [])
        except BulkWriteError as bwe:
            # Ignore duplicate key errors; count successful inserts.
            details = bwe.details or {}
            write_errors = details.get("writeErrors", []) or []
            dup_count = sum(1 for e in write_errors if e.get("code") == 11000)
            inserted = (len(docs_list) - len(write_errors)) if write_errors else 0
            return max(inserted, 0)

    def close(self) -> None:
        """Close the underlying client."""
        try:
            self._client.close()
        except Exception:
            pass

    # --------------- Context manager ---------------

    def __enter__(self) -> "MongoSink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
