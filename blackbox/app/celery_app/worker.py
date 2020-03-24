from .celery import celery, init_celery
from ..app import create_app

flask_app = create_app()
init_celery(celery, flask_app)
