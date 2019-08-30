from celery import Celery
from settings import APP_NAME, BROKER_URL
from blackbox.blackbox import BlackBoxAnomalyDetection

celery_app = Celery(APP_NAME, broker=BROKER_URL)

# Tasks
@celery_app.task
def train_blackbox():
    pass
