# QMem

QMem is a toolkit for vector search.  
It provides a **command-line interface (CLI)** and a **Python library** for interacting with a Qdrant database.  
It is designed for **directness and utility**, offering a guided CLI for interactive tasks and a minimal Python API for programmatic control.

---

## 🚀 Installation

```bash
pip install qmem
```

---

## 🛠️ Commands

### 🔹 init
Initializes the configuration.

**CLI**
```bash
qmem init
```

---

### 🔹 create
Creates a vector collection.

**CLI**
```bash
qmem create
```

**Library**
```python
import qmem

qmem.create(
    collection_name="my-collection",
    dim=1536,
    distance_metric="cosine"
)
```

---

### 🔹 ingest
Ingests data into a collection.

**CLI**
```bash
qmem ingest
```

**Library**
```python
import qmem

qmem.ingest(
    file="path/to/data.jsonl",
    embed_field="text"
)
```

---

### 🔹 retrieve
Performs a vector search.

**CLI**
```bash
qmem retrieve "your query text"
```

**Library**
```python
import qmem

results = qmem.retrieve(
    query="your query text",
    top_k=3
)
print(results)
```

---

### 🔹 index
Creates an index on metadata for filtering.

**CLI**
```bash
qmem index
```

---

### 🔹 filter
Retrieves records by metadata.

**CLI**
```bash
qmem filter
```

**Library**
```python
import qmem

filter_payload = {
  "must": [
    { "key": "genre", "match": { "value": "Sci-Fi" } }
  ]
}

results = qmem.filter(filter_json=filter_payload, limit=10)
print(results)
```

---

### 🔹 retrieve-filter
Combines vector search with metadata filtering.

**CLI**
```bash
qmem retrieve-filter "your query text"
```

**Library**
```python
import qmem

filter_payload = {
  "must": [
    { "key": "genre", "match": { "value": "Sci-Fi" } }
  ]
}

results = qmem.retrieve_filter(
    query="your query text",
    filter_json=filter_payload,
    top_k=2
)
print(results)
```

---

## 📜 License
MIT
