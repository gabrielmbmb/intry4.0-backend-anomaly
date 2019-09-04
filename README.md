# PLATINUM - Blackbox Anomaly Detection

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

This Python package is a 'Blackbox' model that implements several Machine Learning algorithms to flag anomaly points of 
data received from the Orion Context Broker (FIWARE component) and is part of the PLATINUM project.

## Getting started

Install the Python packages that are required:

    pip install -r requirements.txt

## Running the app

Once you have installed all the necessaries Python packages you will have to run the Redis server that is required for
Celery:
    
    chmod +x run-redis.sh
    ./run-redis.sh
    
This will download, compile and run Redis automatically. The next time that the script is executed, the download of
Redis won't be necessary.

Run the Celery worker from parent directory:

    celery -A blackbox.api.async_tasks worker -l info
    
Finally run the APP:

    python main.py
    
## Settings

The app loads the following settings from the environment variables:

* APP_DEBUG
* APP_NAME
* APP_DESC
* APP_VERSION
* APP_HOST
* APP_PORT
* API_ANOMALY_ENDPOINT
* MODELS_ROUTE
* MODELS_ROUTE_JSON
* CELERY_BROKER_URL
* CELERY_RESULT_BACKEND

These variables can be defined in a _**.env**_ inside the parent folder, as follows:

    APP_DEBUG=False
    APP_NAME='PLATINUM - Blackbox Anomaly Detection'
    APP_DESC='A simple API to call the Blackbox Anomaly Detection model.'
    APP_VERSION='1.0'
    APP_HOST='localhost'
    APP_PORT=5678
    API_ANOMALY_ENDPOINT='api/v1/anomaly'
    MODELS_ROUTE='./models'
    MODELS_ROUTE_JSON='./models/models.json'
    CELERY_BROKER_URL='redis://localhost:6379/0'
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'

## Built With

* [Scikit-Learn](https://scikit-learn.org/stable/index.html)
* [Keras](https://keras.io)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Flask-RESTPlus](https://flask-restplus.readthedocs.io/en/stable/)
* [Celery](http://www.celeryproject.org/)

## Authors

* **Gabriel Martín Blázquez**, from [BISITE](bisite.usal.es) - gmartin_b@usal.es
