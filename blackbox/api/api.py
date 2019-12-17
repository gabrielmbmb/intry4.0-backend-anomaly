from flask import Flask
from flask_restplus import Api
from blackbox import settings
from blackbox import version
from blackbox.api.anomaly.namespace import anomaly_ns

# Create Flask App
app = Flask(settings.APP_NAME)

# Wrap Flask App with Flask Restplus
api = Api(
    app,
    version=version.__version__,
    title=settings.APP_NAME,
    description=settings.APP_DESC,
    doc="/swagger",
)

# Add namespaces to the API
api.add_namespace(anomaly_ns)
