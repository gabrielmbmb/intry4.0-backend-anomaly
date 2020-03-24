from celery import Celery


def make_celery():
    """
    Create a Celery app instance.

    Returns:
        celery.app.base.Celery: the Celery app.
    """
    celery = Celery()
    return celery


def init_celery(celery, app):
    """
    Init the configuration of a Celery app from a Flask app.

    Args:
        celery (celery.app.base.Celery): the Celery app.
        app (flask.app.Flask): the Flask app.
    """
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask


celery = make_celery()
