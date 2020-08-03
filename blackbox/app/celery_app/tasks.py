import json
import requests
import pandas as pd
from flask_mongoengine import DoesNotExist
from blackbox.blackbox import BlackBoxAnomalyDetection
from blackbox.app.api.modules.ml.models import BlackboxModel
from .celery import celery, init_celery
from .tasks_names import CELERY_TRAIN_TASK, CELERY_PREDICT_TASK
from ..app import create_app

flask_app = create_app()
init_celery(celery, flask_app)


@celery.task(name=CELERY_TRAIN_TASK, bind=True)
def train_task(self, model_id, data, models_parameters):
    """
    Celery's task which will create the training DataFrame and train a Blackbox model.

    Args:
        model_id (str): Blackbox model id.
        data (dict): containing the keys columns and data.
        models_parameters (dict): each key is a model and has the model parameters.

    Returns:
        dict: status of the task.
    """

    def cb_function(progress, message):
        self.update_state(
            state="PROGRESS",
            meta={"current": progress, "total": 100, "status": message},
        )

    # Create train DataFrame
    df = pd.read_json(json.dumps(data), orient="split")

    # Get Blackbox model info
    try:
        blackbox_model = BlackboxModel.objects.get(model_id=model_id)
    except DoesNotExist:
        return {
            "current": 100,
            "total": 100,
            "status": f"Failure. There is no Blackbox model with f{model_id}",
        }

    # Create Blackbox
    blackbox = BlackBoxAnomalyDetection(verbose=True)

    # Add models to the Blackbox
    for model in models_parameters.keys():
        blackbox.add_model(model, **models_parameters[model])

    # Train the Blackbox model
    blackbox.train_models(df, cb_func=cb_function)

    # Generate pickle of the Blackbox model
    pickle = blackbox.save()

    # Save pickle in MongoDB
    blackbox_model.trained = True
    blackbox_model.saved.put(pickle)
    blackbox_model.save()

    # Webhook
    try:
        requests.post(
            f"{flask_app.config['TRAIN_WEBHOOK']}/{model_id}/train/finished/"
        )
    except requests.exceptions.ConnectionError:
        print("Could not notify the web hook!")
    except requests.exceptions.InvalidURL:
        print("The webhook URL provided is not valid!")

    return {"current": 100, "total": 100, "status": "Train ended"}


@celery.task(name=CELERY_PREDICT_TASK)
def predict_task(model_id, data):
    """
    Celery's task which will predict anomalies in the data provided with a Blackbox
    model.

    Args:
        model_id (str): Blackbox model id.
        data (dict): containing the keys columns and data.

    Returns:
        dict: status of the task.
    """

    # Create DataFrame
    df = pd.read_json(json.dumps(data), orient="split")

    # Get Blackbox model info
    try:
        blackbox_model = BlackboxModel.objects.get(model_id=model_id)
    except DoesNotExist:
        return {
            "current": 100,
            "total": 100,
            "status": f"Failure. There is no Blackbox model with f{model_id}",
        }

    # Retrieve Blackbox model from MongoDB
    blackbox = BlackBoxAnomalyDetection()
    loaded = blackbox_model.saved.read()
    blackbox.load(loaded)

    # Make the predictions
    predictions = blackbox.flag_anomaly(df)

    try:
        response = requests.post(
            f"{flask_app.config['TRAIN_WEBHOOK']}/{model_id}/predict/result/",
            json=predictions
        )
    except requests.exceptions.ConnectionError:
        print("Could not send new predictions to the web hook!")
    except requests.exceptions.InvalidURL:
        print("The webhook URL provided is not valid!")

    if response.status_code != 200:
        print("New predictions not received!")

    return {
        "current": 100,
        "total": 100,
        "status": "Prediction ended",
        "result": predictions
    }
