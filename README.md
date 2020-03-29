# PLATINUM - Blackbox Anomaly Detection

[![Platinum](https://platinum.usal.es/themes/startupgrowth_lite/logo.png)
](https://platinum.usal.es)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Build Status](https://travis-ci.com/gabrielmbmb/platinum-blackbox-anomaly.svg?token=Ym8dypMsw2NFNbxtSMrV&branch=master)](https://travis-ci.com/gabrielmbmb/platinum-blackbox-anomaly)
[![codecov](https://codecov.io/gh/gabrielmbmb/platinum-blackbox-anomaly/branch/master/graph/badge.svg?token=lAfRL6ePBZ)](https://codecov.io/gh/gabrielmbmb/platinum-blackbox-anomaly)
![Black](https://img.shields.io/badge/code%20style-black-000000.svg)

This package is a 'Blackbox' model that implements several anomaly detection algorithms.

## Getting started

### Python installation

Install the Python packages that are required:

    pip install -r requirements.txt

## Running the app

### With Python

Once you have installed all the necessaries Python packages you will have to run the Redis server that is required for
Celery:
  
    chmod +x run-redis.sh
    ./run-redis.sh
  
This will download, compile and run Redis automatically. The next time that the script is executed, the download of
Redis won't be necessary.

Run the Celery worker from parent directory:

    celery worker -A blackbox.app.celery_app.tasks -l info --beat

Finally, run the Flask app with Gunicorn:

    gunicorn --bind 0.0.0.0:5678 -w 4 "blackbox.app.app:create_app()"

### With Docker

The application can be run using the Docker Compose file which includes all the services required:

    docker-compose up

Additionally, the docker image of the Blackbox Anomaly Detection can be build using the following command:

    docker build -t <tag_name_you_like> .

## Settings

The app loads the following settings from the environment variables:

- **MONGODB_DB**: MongoDB database where the API will store the data. Defaults to "blackbox".
- **MONGODB_HOST**: MongoDB host URL. Defaults to "localhost".
- **MONGODB_PORT**: MongoDB port. Defaults to 27017.
- **MONGODB_USERNAME**: MongoDB username in case credentials are required. Defaults to "".
- **MONGODB_PASSWORD**: MongoDB password in case credentials are required. Defaults to "".
- **CELERY_BROKER_URL**: URL of the broker that Celery will use. Defaults to "redis://localhost:6379/0".
- **CELERY_RESULT_BACKEND**: URL of the result backend that Celery will use. Defaults to "redis://localhost:6379/1".
- **CELERY_OCB_PREDICTIONS_FREQUENCY**: frequency (float, seconds) of Celery's periodic task that will try to send the predictions to the OCB in case any prediction could not be sent before. Defaults to "3600.0".
- **TRAIN_WEBHOOK**: Host URL that will be notified when a model has ended its training process. Defaults to "localhost:6789".
- **ORION_HOST**: Orion Context Broker host URL. Defaults to "localhost".
- **ORION_PORT**: Orion Context Broker port. Defaults to "1026".
- **FIWARE_SERVICE**: Orion Context Broker service. Defaults to "blackbox".
- **FIWARE_SERVICEPATH**: Orion Context Broker service path. Defaults to "/".

## Built With

- [Scikit-Learn](https://scikit-learn.org/stable/index.html)
- [Keras](https://keras.io)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [Flask-RESTX](https://flask-restx.readthedocs.io/en/latest/)
- [Gunicorn](https://gunicorn.org/)
- [Celery](http://www.celeryproject.org/)

## Authors

- **Gabriel Martín Blázquez**, from [BISITE](https://bisite.usal.es) - gmartin_b@usal.es
