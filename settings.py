import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

APP_NAME = os.environ.get('APP_NAME')
APP_DESC = os.environ.get('APP_DESC')
APP_VERSION = os.environ.get('APP_VERSION')
MODELS_ROUTE = os.environ.get('MODELS_ROUTE')
MODELS_ROUTE_JSON = os.environ.get('MODELS_ROUTE_JSON')
BROKER_URL = os.environ.get('BROKER_URL')