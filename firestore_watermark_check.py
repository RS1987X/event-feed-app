from google.cloud import firestore

db = firestore.Client(project="event-feed-app-463206")
doc = db.collection("ingest_state").document("gmail").get()
print(doc.to_dict())