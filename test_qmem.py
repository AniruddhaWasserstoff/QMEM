import qmem as qm

# # Create a collection
# qm.create(collection_name="abc", dim=1024, distance_metric="cosine")

# # Ingest data from a fill
# qm.ingest(
#     file="/home/aniruddha/Desktop/qmem/data.jsonl",
#     embed_field="query", # optional, keep these fields in payload
    
# )

#Retrieve results (pretty table by default)
# table = qm.retrieve(
#     query="What is the maximum grace period allowed for premium payment in India?",
#     top_k=100,
#     collection_name="Policy_chunk",   
#     show=["type", "title"]            
# )
# print(table)


# table = qm.retrieve_filter(
#     query="Does SBI Travel Insurance cover lost passport claims?",
#     filter_json=".qmem/filters/latest.json",
#     top_k=10,
#     collection_name="Policy_chunk",
#     show=["type", "title"]
# )
# print(table)

mirrored = qm.mongo(
    collection_name="testing",            # your existing Qdrant collection
    mongo_uri="mongodb://127.0.0.1:27017",     # your local Mongo
    mongo_db="qmem_payload_db",                # DB name you want in Mongo
    mongo_collection="qmem_payload",           # collection name in Mongo
    fields=["description", "title"]
    # batch_size=1000,                         # optional: Qdrant scroll page size
    # max_docs=None,                           # optional: cap how many docs to mirror
)
print("Done")