import os
import sys
from flask import Flask
from blackbox.app.celery_app.celery import celery, init_celery
from blackbox.app.extensions import cors, cache, db, api

CONFIG_NAME_MAPPER = {
    "development": "blackbox.app.config.DevelopConfig",
    "production": "blackbox.app.config.ProductionConfig",
    "testing": "blackbox.app.config.TestingConfig",
}


def create_app(config=None, **kwargs):
    """
    Flask application factory, as explained here:
    https://flask.palletsprojects.com/en/1.1.x/patterns/appfactories/

    Args:
        config (config.BaseConfig): Flask app config. Defaults to
            config.ProductionConfig.

    Returns:
        flask.app.Flask: the Flask app.
    """
    app = Flask(__name__.split(".")[0], **kwargs)

    env_flask_config = os.getenv("FLASK_ENV")
    if not env_flask_config and not config:
        flask_config = "production"
    elif config is None:
        flask_config = env_flask_config
    else:
        if env_flask_config:
            pass

    try:
        app.config.from_object(CONFIG_NAME_MAPPER[flask_config])
    except ImportError:
        app.logger.error(
            f"Invalid Flask configuration: {flask_config}. Rerun the app but first "
            f"define the environment variable FLASK_CONFIG with one of the "
            f"values: {', '.join(CONFIG_NAME_MAPPER.keys())} "
        )
        sys.exit(1)

    app.config.from_object(config)
    register_extensions(app)
    init_celery(celery, app)
    return app


def register_extensions(app):
    """
    Init Flask app extensions.

    Args:
        app (flask.app.Flask): the Flask app.
    """
    cors.init_app(app)
    cache.init_app(app)
    db.init_app(app)
    api.init_app(app, title=app.config["API_NAME"], description=app.config["API_DESC"])
