# QMEM


**Memory library for seamless data ingestion, storage, and retrieval with customizable embedding models.**

QMem is a toolkit for vector search.
It provides a command-line interface (CLI) and a Python library for interacting with a Qdrant database.
It is designed for directness and utility, offering a guided CLI for interactive tasks and a minimal Python API for programmatic control..  


---

## Features

- Store documents in a vector database (Qdrant(`cloud`) or Chroma(`local`))  
- Search text by semantic meaning, not just exact words  
- Apply filters to narrow search results  
- Choose embeddings from OpenAI, Gemini, Voyage, or MiniLM  
- MongoDB integration for analytics  

---

## Installation

```bash
pip install qmem
```

---

## Getting Started

### Step 1: Setup

Run once to configure qmem:

```bash
qmem init
```

You will be asked to select:  
- Database backend: Qdrant (cloud) or Chroma (local)  
- Embedding provider: OpenAI, Gemini, Voyage, or MiniLM  
- API keys and embedding model  

Configuration is saved in `.qmem/config.toml`.

---

### Step 2: Prepare Your Data

qmem expects a `.jsonl` file (one JSON object per line).

Example:

```json
{"query": "What is the plot of Inception?", "response": "A thief enters dreams to steal secrets.", "summary": "Dream heist story.", "title": "Inception", "description": "A skilled thief enters people's dreams to steal corporate secrets.", "genre": "Sci-Fi", "year": 2010}
{"query": "Who is the main character in The Godfather?", "response": "Michael Corleone, son of Vito.", "summary": "Mafia family saga.", "title": "The Godfather", "description": "The patriarch of a crime dynasty hands control to his reluctant son.", "genre": "Crime", "year": 1972}
{"query": "What challenge does Batman face in The Dark Knight?", "response": "He must stop the Joker.", "summary": "Hero vs chaos.", "title": "The Dark Knight", "description": "Batman faces the Joker, a criminal mastermind terrorizing Gotham.", "genre": "Action", "year": 2008}
{"query": "What is Parasite about?", "response": "A poor family cons its way into a rich household.", "summary": "Class divide thriller.", "title": "Parasite", "description": "A poor family infiltrates a wealthy household with dire consequences.", "genre": "Thriller", "year": 2019}
{"query": "What is Interstellar about?", "response": "Explorers travel through a wormhole to save humanity.", "summary": "Space odyssey.", "title": "Interstellar", "description": "A group of explorers journey through a wormhole to find a new home.", "genre": "Adventure", "year": 2014}

```

- `query`: main text to search on  
- `response`:response to the query
- `others`: extra information 


---

### Step 3: Add Data

```python
import qmem as qm

# Create a collection
qm.create(collection_name="testing", dim=1024, distance_metric="cosine")

# Ingest data from file
qm.ingest(file="data.jsonl", embed_field="query",payload_field="x,y,z") #what to put in payload , not added then everything goes in payload
```

---

### Step 4: Search

```python
results = qm.retrieve(
    query="who is batman",
    top_k=5,
    show=["description", "title"]
)
print(results)
```

Results are returned in a table showing similarity scores and payload fields.

---

### Step 5: Use Filters

Build filters interactively (`CLI`):

```bash
qmem filter
```

Filters are saved under `.qmem/filters/`.

Apply a filter in Python:

```python
filtered = qm.retrieve_filter(
    query="who is batman",
    filter_json=".qmem/filters/latest.json",
    top_k=5,
    show=["description", "title"]
)
print(filtered)
```

---

### Step 6: Mirror Data to MongoDB

```python
qm.mongo(
    collection_name="testing",
    mongo_db="final_test_db",
    mongo_collection="final_test",
    fields=["description", "title"]
)
```

This copies your Qdrant collection into MongoDB for downstream use.

---

## CLI Commands

```bash
qmem init      # configure qmem
qmem filter    # create filters interactively
qmem index     # (only for Qdrant) build indices for faster filtering
```

---

## Python API

- `qm.create()` → Create a new collection  
- `qm.ingest()` → Add documents from a file  
- `qm.retrieve()` → Search documents by meaning  
- `qm.retrieve_filter()` → Search with filters  
- `qm.mongo()` → Copy a collection into MongoDB  

---

## Why use qmem?

- Simple to set up and use  
- Works with both local (Chroma) and cloud (Qdrant) databases  
- AI embeddings understand meaning, not just exact words  
- Filters allow more precise retrieval  
- MongoDB integration enables analytics pipelines  

---

## Example Workflow

1. `qmem init` → configure backend and embeddings  
2. `qm.create(...)` → create a collection  
3. `qm.ingest(...)` → add data from JSONL  
4. `qm.retrieve(...)` → search data by meaning  
5. `qmem filter` + `qm.retrieve_filter(...)` → apply filters  
6. `qm.mongo(...)` → mirror to MongoDB if needed  

---

qmem makes it easy to:  
- Store your documents  
- Search them intelligently  
- Filter results by properties  
- Extend usage with MongoDB integration  

Searching for “car” may also return results for “automobile” or “vehicle.”