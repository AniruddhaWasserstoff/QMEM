import qmem as qm
qm.create(collection_name="abc", dim=1024, distance_metric="cosine")

qm.ingest(
    file="/home/aniruddha/Desktop/qmem/data.jsonl",
    embed_field="query",
    # payload_field="query,response,genre,year" # optional: restrict payload keys
    
)

table = qm.retrieve(query="dream heist", top_k=3)  # pretty table by default
print(table)

filter_json = {
  "must": [
    {
      "key": "genre",
      "match": {
        "value": "Sci-Fi"
      }
    }
  ]
}

table = qm.filter(filter_json=filter_json, limit=5)
print(table)


table = qm.retrieve_filter(
    query="space travel",
    filter_json=latest.json,
    top_k=3,
)
print(table)
