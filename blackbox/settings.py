import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

APP_DEBUG = 'True' == os.getenv('APP_DEBUG', 'False')
APP_NAME = 'PLATINUM - Blackbox Anomaly Detection'
APP_DESC = 'A simple API to call the Blackbox Anomaly Detection model.'
APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
APP_PORT = os.getenv('APP_PORT', '5678')
API_ANOMALY_ENDPOINT = os.getenv('API_ANOMALY_ENDPOINT', 'api/v1')
MODELS_ROUTE = os.getenv('MODELS_ROUTE', os.path.expanduser('~/blackbox/models'))
MODELS_ROUTE_JSON = os.getenv('MODELS_ROUTE_JSON', os.path.join(MODELS_ROUTE, 'models.json'))
MODELS_ROUTE_TRASH = os.getenv('MODELS_ROUTE_TRASH', os.path.join(MODELS_ROUTE, 'trash'))
CELERY_BROKER_URL = os.getenv('BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
ORION_CONTEXT_BROKER = os.getenv('ORION_CONTEXT_BROKER', 'http://localhost:1026')
FIWARE_SERVICEPATH = os.getenv('FIWARE_SERVICEPATH', '/')
FIWARE_SERVICE = os.getenv('FIWARE_SERVICE', '')
