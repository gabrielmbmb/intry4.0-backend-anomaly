import os


class BaseConfig(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = "some secret key here"
    SERVER_IP = os.getenv("SERVER_IP", "localhost")

    # API
    API_NAME = "InTry 4.0 - Blackbox Anomaly Detection"
    API_DESC = "An API to call the Blackbox Anomaly Detection model"

    # Cache
    CACHE_TYPE = "simple"
    CACHE_DEFAULT_TIMEOUT = 300

    # CORS
    CORS_ORIGIN_WHITELIST = "*"

    # RESTX
    RESTX_MASK_SWAGGER = False

    # MongoDB
    MONGODB_CONNECT = False
    MONGODB_DB = os.getenv("MONGODB_DB", "blackbox")
    MONGODB_HOST = os.getenv("MONGODB_HOST", "localhost")
    MONGODB_PORT = int(os.getenv("MONGODB_PORT", 27017))
    MONGODB_USERNAME = os.getenv("MONGODB_USERNAME", "")
    MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD", "")

    # Celery
    BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv(
        "CELERY_RESULT_BACKEND", "redis://localhost:6379/1"
    )
    CELERY_OCB_PREDICTIONS_FREQUENCY = float(
        os.getenv("CELERY_OCB_PREDICTIONS_FREQUENCY", "3600.0")
    )

    # Train ended webhook
    TRAIN_WEBHOOK = os.getenv("TRAIN_WEBHOOK", "localhost:6789")

    # Orion Context Broker
    ORION_HOST = os.getenv("ORION_HOST", "localhost")
    ORION_PORT = os.getenv("ORION_PORT", "1026")
    FIWARE_SERVICE = os.getenv("FIWARE_SERVICE", "blackbox")
    FIWARE_SERVICEPATH = os.getenv("FIWARE_SERVICEPATH", "/")


class DevelopConfig(BaseConfig):
    ENV = "development"
    DEBUG = True
    PROPAGATE_EXCEPTIONS = False


class ProductionConfig(BaseConfig):
    ENV = "production"


class TestingConfig(BaseConfig):
    ENV = "testing"
    TESTING = True
    DEBUG = True
