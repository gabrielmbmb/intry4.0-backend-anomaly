# PLATINUM - Blackbox Anomaly Detection

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Build Status](https://travis-ci.com/gabrielmbmb/platinum-blackbox-anomaly.svg?token=Ym8dypMsw2NFNbxtSMrV&branch=master)](https://travis-ci.com/gabrielmbmb/platinum-blackbox-anomaly)
[![codecov](https://codecov.io/gh/gabrielmbmb/platinum-blackbox-anomaly/branch/master/graph/badge.svg?token=lAfRL6ePBZ)](https://codecov.io/gh/gabrielmbmb/platinum-blackbox-anomaly)

This package is a 'Blackbox' model that implements several algorithms to flag as anomalous or not the
data received from the Orion Context Broker (FIWARE component) and is part of the PLATINUM project.

## Getting started

### Python installation

Install the Python packages that are required:

    pip install -r requirements.txt

Then, you can install the PLATINUM - Blackbox Anomaly Detection package:

    pip install .

Alternatively, you can install the package with setuptools:

    python setup.py install --record platinum-blackbox-files.txt
    
If you installed the package with pip and you want to uninstall it:

    pip uninstall platinum-anomaly-detection
    
If you installed the package with setuptools:

    xargs rm -rf < platinum-blackbox-files.txt
    
## Running the app

### With Python

Once you have installed all the necessaries Python packages you will have to run the Redis server that is required for
Celery:
    
    chmod +x run-redis.sh
    ./run-redis.sh
    
This will download, compile and run Redis automatically. The next time that the script is executed, the download of
Redis won't be necessary.

Run the Celery worker from parent directory:

    celery -A blackbox.tasks.tasks worker -l info
    
Finally run the APP:

    blackbox
    
### With Docker

The application can be run using the Docker Compose file which includes all the services required.

    docker-compose up
    
Additionally, the docker image of the Blackbox Anomaly Detection can be build using the following command:

    docker build -t <tag_name_you_like> .
    
## Settings

The app loads the following settings from the environment variables:

* APP_DEBUG: indicates if the app will be run in debug mode. Defaults to False.
* APP_NAME: the app name that will be shown in Swagger API description. 
            Defaults to 'PLATINUM - Blackbox Anomaly Detection'.
* APP_DESC: the app description that will be shown in Swagger API description.
* APP_HOST: the hostname where the APP will be listening for requests. Defaults to 'localhost'.
* APP_PORT: the APP port. Defaults to 5678.
* API_ANOMALY_ENDPOINT: the Anomaly Detection API Endpoint. Defaults to 'api/v1/anomaly'
* MODELS_ROUTE: the path where all the models will be saved. Defaults to './models'
* MODELS_ROUTE_JSON: the path where the JSON file storing information about the models and entities will be saved.
                     Defaults to './models/models.json'.
* CELERY_BROKER_URL: the URL of the broker that Celery will use. Defaults to 'redis://localhost:6379/0'.
* CELERY_RESULT_BACKEND: the URL of the backend that Celery will use. Defaults to 'redis://localhost:6379/0'.
* ORION_CONTEXT_BROKER: the URL of the Orion Context Broker.
* FIWARE_SERVICEPATH: Orion Context Broker service path
* FIWARE_SERVICE: Orion Context Broker name service

These variables can be defined in a _**.env**_ inside the parent folder, as follows:

    APP_DEBUG=False
    APP_HOST='0.0.0.0'
    APP_PORT=5678
    API_ANOMALY_ENDPOINT='api/v1'
    MODELS_ROUTE='./models'
    MODELS_ROUTE_JSON='./models/models.json'
    CELERY_BROKER_URL='redis://localhost:6379/0'
    CELERY_RESULT_BACKEND='redis://localhost:6379/0'
    ORION_CONTEXT_BROKER='http://localhost'
    FIWARE_SERVICEPATH='/'    
    FIWARE_SERVICE='' 
    
## Example
### Training a model for a Orion Context Broker (FIWARE) Entity.

The dataset used for testing is the [NASA Bearing Dataset](http://data-acoustics.com/measurements/bearing-faults/bearing-4/).

The first thing to do is to create an entity in the Orion Context Broker (FIWARE). The communications with the Orion
Context Broker are done via HTTP request. **Postman** or **curl** can be used for this purpose.

To make things easier, a collection of Postman has been created with all the necessary API calls. 

[![Run in Postman](https://run.pstmn.io/button.svg)](https://app.getpostman.com/run-collection/0273ca5f0a6c1602a828).

Creating an entity in Orion Context Broker (FIWARE):

    curl -X POST \
      http://localhost:1026/v2/entities \
      -H 'Content-Type: application/json' \
      -H 'fiware-service: ' \
      -H 'fiware-servicepath: /' \
      -d '{
        "id": "urn:ngsi-ld:Machine:001",
        "type": "Machine",
        "name": {
            "type": "Text",
            "value": "Pressure Machine"
        },
        "Bearing1": {
            "type": "String",
            "value": "0"
        },
        "Bearing2": {
            "type": "String",
            "value": "0"
        },
        "Bearing3": {
            "type": "String",
            "value": "0"
        },
        "Bearing4": {
            "type": "String",
            "value": "0"
        }
    }'
      
The next thing to do is to create the entity in the Anomaly Detection API. The *id* has to be exactly the same as that of
the entity we have created in Orion Context Broker or it can be a regular expression such as _"urn:ngsi-ld:Machine:[0-9]{3}$"_. 
This id will be specified in the end of the URL. The payload will indicate the attributes that will be using to train the 
Anomaly Detection Model. The name of these attributes has to be exactly the same as those of the entity we have created 
in Orion Context Broker:

    curl -X POST \
      'http://localhost:5678/api/v1/entities/urn:ngsi-ld:Machine:[0-9]{3}$' \
      -H 'Content-Type: application/json' \
      -d '{
        "attrs": [
            "Bearing1",
            "Bearing2",
            "Bearing3",
            "Bearing4"
        ]
    }'
        
Response:

    {"message":"The entity urn:ngsi-ld:Machine:001 was created"}

Once the entity has been created in the Anomaly Detection API, a model can be trained. A CSV is provided in this repository
with training data. It can be found in _test_csv/no_anomaly_data.csv_:

    curl -X POST \
      'http://localhost:5678/api/v1/train/urn:ngsi-ld:Machine:[0-9]{3}$' \
      -H 'content-type: multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW' \
      -F file=@/home/gmdev/PycharmProjects/PLATINUM-blackbox/test_csv/no_anomaly_data.csv \
      -F 'input_arguments=Bearing1,Bearing2,Bearing3,Bearing4' \
      -F name=test_model
       
Response:

    {
        "message":"The file was train_data.csv uploaded. Training model for entity urn:ngsi-ld:Machine:001",
        "task_status_url":"http://localhost:5678/api/v1/anomaly/task/96695519-97c7-4c3e-9a06-fb720065c798"
    }
        
An URL will be provided in which the progress of the training progress can be seen.

    curl -X GET http://localhost:5678/api/v1/task/fc2f6cfe-b6f0-4caf-aad8-012e62ada7d6
        
Response:

    {
        "current":100,
        "state":"SUCCESS",
        "status":"TASK ENDED",
        "total":100
    }

The next thing is to create a subscription in Orion Context Broker, so the Anomaly Detection API
will receive data when the value of an attribute changes:

    curl -X POST \
      http://localhost:1026/v2/subscriptions/ \
      -H 'Content-Type: application/json' \
      -H 'fiware-service: ' \
      -H 'fiware-servicepath: /' \
      -d '{
        "description": "Notify Anomaly Prediction API of changes in urn:ngsi-ld:Machine:*",
        "subject": {
            "entities": [
                {
                    "idPattern": "urn:ngsi-ld:Machine:[0-9]{3}$"
                }
            ],
            "condition": {
              "attrs": [
                "Bearing1",
                "Bearing2",
                "Bearing3",
                "Bearing4"
              ]
            }
            },
            "notification": {
            "http": {
              "url": "http://blackbox:5678/api/v1/predict"
            },
            "attrs": [
                "Bearing1",
                "Bearing2",
                "Bearing3",
                "Bearing4"
            ],
            "metadata": ["dateCreated", "dateModified"]
        }
    }'
    
We can update attributes of the entity manually in order to see the Blackbox operation with the following command:

    curl -X POST \
      http://localhost:1026/v2/entities/urn:ngsi-ld:Machine:001/attrs \
      -H 'Content-Type: application/json' \
      -H 'fiware-service: ' \
      -H 'fiware-servicepath: /' \
      -d '{
        "Bearing1": {
            "type": "String",
            "value": "0.06228134765624953"
        },
        "Bearing2": {
            "type": "String",
            "value": "0.075808203125"
        },
        "Bearing3": {
            "type": "String",
            "value": "0.08437338867187452"
        },
        "Bearing4": {
            "type": "String",
            "value": "0.044273291015623564"
        }
    }'
        
Orion Context Broker will send a POST to the specified URL with the new values of the attrs specified in the subscription. 
When the Anomaly Detection API receives this POST, a prediction will be made automatically and send back to OCB, where 
the predictions will be saved as attributes of the entity. We can see the predictions with the following command:

    curl -X GET \
      http://localhost:1026/v2/entities/urn:ngsi-ld:Machine:001 \
      -H 'fiware-service: ' \
      -H 'fiware-servicepath: /'
      
Response:

    {
        "id": "urn:ngsi-ld:Machine:001",
        "type": "Machine",
        "AnomalyAutoencoder": {
            "type": "Boolean",
            "value": "False",
            "metadata": {}
        },
        "AnomalyGaussianDistribution": {
            "type": "Boolean",
            "value": "False",
            "metadata": {}
        },
        "AnomalyIsolationForest": {
            "type": "Boolean",
            "value": "False",
            "metadata": {}
        },
        "AnomalyKMeans": {
            "type": "Boolean",
            "value": "False",
            "metadata": {}
        },
        "AnomalyOneClassSVM": {
            "type": "Boolean",
            "value": "False",
            "metadata": {}
        },
        "AnomalyPCAMahalanobis": {
            "type": "Boolean",
            "value": "False",
            "metadata": {}
        },
        "Bearing1": {
            "type": "String",
            "value": "0.06228134765624953",
            "metadata": {}
        },
        "Bearing2": {
            "type": "String",
            "value": "0.075808203125",
            "metadata": {}
        },
        "Bearing3": {
            "type": "String",
            "value": "0.08437338867187452",
            "metadata": {}
        },
        "Bearing4": {
            "type": "String",
            "value": "0.044273291015623564",
            "metadata": {}
        },
        "name": {
            "type": "Text",
            "value": "Pressure Machine",
            "metadata": {}
        }
    }
    
## Built With

* [Scikit-Learn](https://scikit-learn.org/stable/index.html)
* [Keras](https://keras.io)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Flask](https://flask.palletsprojects.com/en/1.1.x/)
* [Flask-RESTPlus](https://flask-restplus.readthedocs.io/en/stable/)
* [Celery](http://www.celeryproject.org/)

## Authors

* **Gabriel Martín Blázquez**, from [BISITE](https://bisite.usal.es) - gmartin_b@usal.es
