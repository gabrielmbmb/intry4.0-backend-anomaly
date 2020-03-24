from flask_restx import Api
from .modules.ml.namespace import ml_ns
from blackbox import version


# Namespace aggregator
api = Api(doc="/docs", version=version.__version__)

api.add_namespace(ml_ns)
