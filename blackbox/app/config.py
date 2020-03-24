
class BaseConfig(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = "some secret key here"

    # API
    API_NAME = "PLATINUM - Blackbox Anomaly Detection"
    API_DESC = "An API to call the Blackbox Anomaly Detection model"

    # Cache
    CACHE_TYPE = 'simple'
    CACHE_DEFAULT_TIMEOUT = 300

    # CORS
    CORS_ORIGIN_WHITELIST = "*"

    # RESTX
    RESTX_MASK_SWAGGER = False

    # MongoDB
    MONGODB_DB = "blackbox"
    MONGODB_HOST = "localhost"
    MONGODB_PORT = 27017
    MONGODB_USERNAME = ""
    MONGODB_PASSWORD = ""

    # Celery
    BROKER_URL = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND = "redis://localhost:6379/1"

    # Models storing
    BLACKBOX_MODELS_PATH = "./models"


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
