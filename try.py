# test_like_example.py
import json
import os
import time

import qmem as qm  # your high-level API

# ---------- Config ----------
COLLECTION = "testing"
DATA_FILE = "/home/aniruddha/Desktop/qmem/data.jsonl"
FILTER_FILE = ".qmem/filters/latest.json"
QUERY = "who is batman"
PARAPHRASE = "tell me about batman"
TOP_K = 5

print("=== QMem Cache Test (Redis) ===")
print(f"QMEM_CACHE_BACKEND = {os.getenv('QMEM_CACHE_BACKEND', '(not set)')}")
print(f"QMEM_REDIS_URL     = {os.getenv('QMEM_REDIS_URL', 'redis://127.0.0.1:6379/0')}")

# ---------- 1) Create collection ----------
print("\n[1] Create/ensure collection...")
# matches your style: collection_name + distance_metric
qm.create(collection_name=COLLECTION, dim=1024, distance_metric="cosine")
print("    OK")

# ---------- 2) Ingest ----------
print("\n[2] Ingest data from file...")
qm.ingest(
    collection_name=COLLECTION,
    file=DATA_FILE,
    embed_field="query",
)
print("    Ingested (idempotent on repeat)")

# ---------- 3) Retrieve #1 (MISS -> DB, caches set) ----------
print("\n[3] Retrieve #1 (expect MISS_DB)")
t0 = time.time()
table1 = qm.retrieve(
    query=QUERY,
    top_k=TOP_K,
    collection_name=COLLECTION,
    show=["description", "title"],
)
t1 = time.time() - t0
print(table1)   # pretty table string
print(f"    Time: {t1:.3f}s")

# ---------- 4) Retrieve #2 same query (HIT_EXACT from Redis) ----------
print("\n[4] Retrieve #2 same query (expect HIT_EXACT)")
t0 = time.time()
table2 = qm.retrieve(
    query=QUERY,
    top_k=TOP_K,
    collection_name=COLLECTION,
    show=["description", "title"],
)
t2 = time.time() - t0
print(table2)
print(f"    Time: {t2:.3f}s")
if t2 < t1:
    print("    ✅ Faster on second call → likely exact-key cache HIT (Redis)")
else:
    print("    ⚠️ Not faster; timings vary, but second call should avoid DB if cache is active.")

# ---------- 5) Retrieve paraphrase (semantic cache candidate) ----------
print("\n[5] Retrieve paraphrase (semantic cache test)")
t0 = time.time()
table3 = qm.retrieve(
    query=PARAPHRASE,
    top_k=TOP_K,
    collection_name=COLLECTION,
    show=["description", "title"],
)
t3 = time.time() - t0
print(table3)
print(f"    Time: {t3:.3f}s")
print("    Note: Semantic HIT depends on your data and similarity threshold (~0.97).")

from qmem.cache import make_exact_key
import os, redis
ek, _ = make_exact_key(collection="testing", query="who is batman", top_k=5, filt=None)
r = redis.from_url(os.getenv("QMEM_REDIS_URL", "redis://127.0.0.1:6379/0"), decode_responses=True)
print("Redis has exact-key?", bool(r.exists(f"qmem:cache:{ek}")))


print("\n=== Done ===")
