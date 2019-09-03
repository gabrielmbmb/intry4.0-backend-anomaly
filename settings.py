import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

APP_NAME = os.environ.get('APP_NAME') or 'PLATINUM - Blackbox Anomaly Detection'
APP_DESC = os.environ.get('APP_DESC') or 'A simple API to call the Blackbox Anomaly Detection model.'
APP_VERSION = os.environ.get('APP_VERSION') or '1.0'
APP_HOST = os.environ.get('APP_HOST') or 'localhost'
APP_PORT = os.environ.get('APP_PORT') or '5678'
API_ANOMALY_ENDPOINT = os.environ.get('API_ANOMALY_ENDPOINT') or 'api/v1/anomaly'
MODELS_ROUTE = os.environ.get('MODELS_ROUTE') or './models'
MODELS_ROUTE_JSON = os.environ.get('MODELS_ROUTE_JSON') or os.path.join(MODELS_ROUTE, 'models.json')
CELERY_BROKER_URL = os.environ.get('BROKER_URL') or 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0'
