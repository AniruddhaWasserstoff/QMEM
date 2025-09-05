import qmem as qm

# Create a collection
qm.create(collection_name="abc", dim=1024, distance_metric="cosine")

# Ingest data from a fill
qm.ingest(
    file="/home/aniruddha/Desktop/qmem/data.jsonl",
    embed_field="query", # optional, keep these fields in payload
    
)

# Retrieve results (pretty table by default)
table = qm.retrieve(query="list customers", top_k=5)
print(table)