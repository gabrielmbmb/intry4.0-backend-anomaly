import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

APP_NAME = os.environ.get('APP_NAME') or 'PLATINUM - Blackbox Anomaly Detection'
APP_DESC = os.environ.get('APP_DESC') or 'A simple API to call the Blackbox Anomaly Detection model.'
APP_VERSION = os.environ.get('APP_VERSION') or '1.0'
MODELS_ROUTE = os.environ.get('MODELS_ROUTE') or './models'
MODELS_ROUTE_JSON = os.environ.get('MODELS_ROUTE_JSON') or os.path.join(MODELS_ROUTE, 'models.json')
BROKER_URL = os.environ.get('BROKER_URL') or 'amqp://localhost'
