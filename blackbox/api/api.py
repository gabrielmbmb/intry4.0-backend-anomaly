from flask import Flask
from flask_restplus import Api
from blackbox import settings
from blackbox import version
from blackbox.utils.orion import check_orion_connection
from blackbox.api.anomaly.anomaly import anomaly_ns

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


def run_api():
    """Runs the API with the configuration inside the config file"""
    if not check_orion_connection():
        print("Unable to connect to Orion Context Broker...")
        print("Blackbox will continue its execution anyway...")
    else:
        print("Orion Context Broker is up")

    if settings.API_SSL:
        app.run(
            host=settings.APP_HOST,
            port=settings.APP_PORT,
            debug=settings.APP_DEBUG,
            ssl_context="adhoc",
        )
    else:
        app.run(
            host=settings.APP_HOST, port=settings.APP_PORT, debug=settings.APP_DEBUG
        )
