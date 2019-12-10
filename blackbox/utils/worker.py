from blackbox.settings import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from celery import Celery

celery_app = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
