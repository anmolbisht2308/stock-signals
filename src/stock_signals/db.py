from pymongo import MongoClient
from pymongo.database import Database

from stock_signals.config import Settings


def connect(settings: Settings) -> Database:
    client: MongoClient = MongoClient(settings.mongo_uri, tz_aware=True)
    return client[settings.mongo_db]
