# qmem 

`qmem`  for **easy ingestion and retrieval** with embeddings using qdrant
Supports both **CLI** and **Python API**.

---

## 🚀 Installation

```bash
pip install -e .
```

---

## ⚙️ CLI Usage

### 1. Init (configure keys & embedding model)

```bash
qmem init
```

### 2. Ingest data

```bash
qmem ingest
```

You’ll be prompted for:
- **collection_name**
- **data file path** (JSON or JSONL)
- **field to embed** (e.g. `query`, `response`, `sql_query`, `doc_id`)
- **payload fields** (comma-separated, leave empty to keep all)

### 3. Retrieve results

```bash
qmem retrieve
```

You’ll be prompted for:
- **collection_name**
- **query**
- **top_k** (number of results to return)

---

## 🐍 Python API

```python
import qmem as qm

# Create a collection
qm.create(collection_name="test_learn", dim=1536, distance_metric="cosine")

# Ingest data from a file
qm.ingest(
    file="/home/User/data.jsonl",
    embed_field="sql_query",
    payload_field="query,response",  # optional, keep these fields in payload
)

# Retrieve results (pretty table by default)
table = qm.retrieve(query="list customers", top_k=5)
print(table)
```

---

## 📄 License

MIT
