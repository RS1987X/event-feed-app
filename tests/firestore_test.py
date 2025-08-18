from google.cloud import firestore

PROJECT_ID = "event-feed-app-463206"
db = firestore.Client(project=PROJECT_ID)

doc_ref = db.collection("ingest_state").document("smoke_test")
doc_ref.set({"hello": "world"})
print("Wrote doc")

doc = doc_ref.get()
print("Read back:", doc.to_dict())