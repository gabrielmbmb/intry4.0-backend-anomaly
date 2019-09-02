from celery import Celery
from settings import APP_NAME, CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from blackbox.blackbox import BlackBoxAnomalyDetection

celery_app = Celery(APP_NAME, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# Tasks
@celery_app.task
def train_blackbox():
    pass
