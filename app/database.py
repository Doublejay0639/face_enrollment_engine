import os
from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

mongo_uri = os.getenv("MONGO_URI")


if not mongo_uri:
    raise RuntimeError("MONGO_URI is missing from the environment")

client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)

db = client["test"]
enrollments = db["biometric_enrollments"]
enrollments.create_index("matricNumber", unique=True)
