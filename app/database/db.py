from pymongo import MongoClient
MONGO_URI = "mongodb+srv://pratiksha:pratiksha1902@explainable-ml.zlnjqoi.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)
db = client["explainable_ml"]
collection = db["predictions"]

print("Connecting to MongoDB...")

try:
    print(client.list_database_names())
    print("MongoDB Connected ✅")
except Exception as e:
    print("MongoDB Error ❌:", e)

collection.insert_one({"test": "working"})
print("Inserted test document")