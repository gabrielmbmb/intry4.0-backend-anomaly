from flask_cors import CORS
from flask_caching import Cache
from flask_mongoengine import MongoEngine
from .api.api import api

cors = CORS()
cache = Cache()
db = MongoEngine()
api = api
