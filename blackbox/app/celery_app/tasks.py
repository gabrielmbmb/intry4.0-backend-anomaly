import json
import requests
import pandas as pd
from flask_mongoengine import DoesNotExist
from blackbox.blackbox import BlackBoxAnomalyDetection
from blackbox.app.api.modules.ml.models import BlackboxModel, BlackboxPrediction
from .celery import celery, init_celery
from .tasks_names import CELERY_TRAIN_TASK, CELERY_PREDICT_TASK
from ..app import create_app

flask_app = create_app()
init_celery(celery, flask_app)


@celery.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    sender.add_periodic_task(
        flask_app.config["CELERY_OCB_PREDICTIONS_FREQUENCY"],
        send_predictions_to_ocb.s(),
    )


@celery.task
def send_predictions_to_ocb():
    """
    Celery's periodic task to try to send past predictions (that could not be sent in
    the moment the prediction was made) to Orion Context Broker.
    """

    blackbox_predictions = BlackboxPrediction.objects()

    if blackbox_predictions:
        print("Sending past predictions to OCB...")

        entities = []
        for prediction in blackbox_predictions:
            entity_info = {
                "id": f"urn:ngsi-ld:BlackboxModel:{prediction.model.model_id}",
                "type": "BlackboxModel",
            }
            entities.append({**entity_info, **prediction.predictions})

        url = (
            f"http://{flask_app.config['ORION_HOST']}:{flask_app.config['ORION_PORT']}"
            f"/v2/op/update"
        )

        headers = {
            "fiware-service": flask_app.config["FIWARE_SERVICE"],
            "fiware-servicepath": flask_app.config["FIWARE_SERVICEPATH"],
            "Content-Type": "application/json",
        }

        data = {"actionType": "APPEND", "entities": entities}

        try:
            requests.post(url, headers, json=data)
            blackbox_predictions.objects().delete()
        except requests.exceptions.RequestException:
            print("Could not sent the past predictions to OCB...")


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
    blackbox_model.saved = pickle
    blackbox_model.save()

    # Webhook
    try:
        requests.post(
            f"http://{flask_app.config['TRAIN_WEBHOOK']}/{model_id}/train/finished/"
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
    blackbox.load(blackbox_model.saved)

    # Make the predictions
    predictions = blackbox.flag_anomaly(df)
    predictions = {
        key: {"type": "Array", "value": value.tolist()}
        for key, value in predictions.items()
    }

    # Send it to OCB
    url = (
        f"http://{flask_app.config['ORION_HOST']}:{flask_app.config['ORION_PORT']}"
        f"/v2/entities"
    )

    headers = {
        "fiware-service": flask_app.config["FIWARE_SERVICE"],
        "fiware-servicepath": flask_app.config["FIWARE_SERVICEPATH"],
        "Content-Type": "application/json",
    }

    entity_id = f"urn:ngsi-ld:BlackboxModel:{model_id}"
    data = {"id": entity_id, "type": "BlackboxModel"}
    data = {**data, **predictions}

    try:
        request = requests.post(url, headers=headers, json={**data, **predictions})

        # entity is already created, update it
        if request.status_code == 422:
            print(
                f"Entity {entity_id} already exists in OCB. Updating it values instead..."
            )
            url = (
                f"http://{flask_app.config['ORION_HOST']}:{flask_app.config['ORION_PORT']}"
                f"/v2/entities/{entity_id}/attrs"
            )
            requests.post(url, headers=headers, json=predictions)

    except requests.exceptions.RequestException:
        # If the predictions could not be sent to the OCB, store it in MongoDB
        print(
            "Could not send the predictions to Orion Context Broker... Storing it in "
            "MongoDB"
        )
        blackbox_prediction = BlackboxPrediction(
            predictions=predictions, model=blackbox_model
        )
        blackbox_prediction.save()

    return {"current": 100, "total": 100, "status": "Prediction ended", "result": data}
