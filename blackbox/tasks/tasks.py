import numpy as np
import pandas as pd
from blackbox import settings
from celery import Celery
from blackbox.utils.api import add_model_entity_json
from blackbox.blackbox import BlackBoxAnomalyDetection
from blackbox.models.unsupervised import (
    AnomalyPCAMahalanobis,
    AnomalyAutoencoder,
    AnomalyKMeans,
    AnomalyIsolationForest,
    AnomalyGaussianDistribution,
    AnomalyOneClassSVM,
    AnomalyLOF,
    AnomalyKNN,
)
from blackbox.utils.csv import CSVReader
from blackbox.utils.orion import update_entity_attrs

# Todo: add logging to tasks

celery_app = Celery(
    "tasks", broker=settings.CELERY_BROKER_URL, backend=settings.CELERY_RESULT_BACKEND
)


# Tasks
@celery_app.task(name="tasks.train", bind=True)
def train_blackbox(
    self, entity_id, filename, model_name, models, input_arguments, additional_params
):
    """
    Celery's task which will read the training data and train the Blackbox model for the
    specified entity.

    Args:
        self (object): Celery app object.
        entity_id (str): Orion Context Broker (FIWARE component) entity ID
        filename (str): path of the file which will be used for training the model.
        model_name (str): model name which will be used to refer to the Blackbox trained
            model.
        models (list of str): list of strings containing the name of anomaly prediction
            models that are going to be used.
        input_arguments (list of str): name of the inputs variables of the Blackbox
            model.
        additional_params (dict): dictionary containing additional params to specify
            to each anomaly detection model.

    Returns:
        dict: status of the task.
    """

    def cb_function(progress, message):
        self.update_state(
            state="PROGRESS",
            meta={"current": progress, "total": 100, "status": message},
        )

    # Read CSV file
    reader = CSVReader(filename)
    data = reader.get_df()

    # Create Blackbox
    blackbox = BlackBoxAnomalyDetection(verbose=True)

    # Add models to the Blackbox
    for anomaly_model_name in BlackBoxAnomalyDetection.AVAILABLE_MODELS:
        params = additional_params[anomaly_model_name]

        if anomaly_model_name == "PCAMahalanobis":
            blackbox.add_model(AnomalyPCAMahalanobis(**params))

        elif anomaly_model_name == "Autoencoder":
            blackbox.add_model(AnomalyAutoencoder(**params))

        elif anomaly_model_name == "KMeans":
            blackbox.add_model(AnomalyKMeans(**params))

        elif anomaly_model_name == "OneClassSVM":
            blackbox.add_model(AnomalyOneClassSVM(**params))

        elif anomaly_model_name == "GaussianDistribution":
            blackbox.add_model(AnomalyGaussianDistribution(**params))

        elif anomaly_model_name == "IsolationForest":
            blackbox.add_model(AnomalyIsolationForest(**params))

        elif anomaly_model_name == "KNearestNeighbors":
            blackbox.add_model(AnomalyKNN(**params))

        elif anomaly_model_name == "LocalOutlierFactor":
            blackbox.add_model(AnomalyLOF(**params))

    # Train the models in Blackbox
    blackbox.train_models(data, cb_func=cb_function)

    # Save the Blackbox
    model_path = settings.MODELS_ROUTE + "/" + entity_id + "/" + model_name + ".pkl"
    blackbox.save_blackbox(model_path)

    # Add the Blackbox to the JSON
    add_model_entity_json(
        settings.MODELS_ROUTE_JSON,
        entity_id,
        model_name,
        model_path,
        filename,
        models,
        input_arguments,
    )

    return {"current": 100, "total": 100, "status": "TASK ENDED"}


@celery_app.task(name="tasks.predict")
def predict_blackbox(entity_id, date, model_path, predict_data):
    """
    Flag a data point received from Orion Context Broker (FIWARE component) as an anomaly or not using an already
    trained model loaded from a pickle file.

    Args:
        entity_id (str): Orion Context Broker (FIWARE component) entity ID
        date (str): date of the new received data.
        model_path (str): path of the trained model.
        predict_data (list): values to predict.

    Returns:
        dict: status of the task with the results of the prediction.
    """
    model = BlackBoxAnomalyDetection(verbose=True)
    model.load_blackbox(model_path)

    predict_data = np.array([predict_data])
    predictions = model.flag_anomaly(predict_data)

    predictions = [str(pred) for pred in predictions[0]]  # transform bool to str
    results = {"entity_id": entity_id, "date": date, "models_predictions": {}}
    attrs = {}

    for n_model, model in enumerate(model.models.items()):
        results["models_predictions"][model[0]] = predictions[n_model]
        attrs[model[0]] = {"type": "Boolean", "value": predictions[n_model]}

    response = update_entity_attrs(entity_id, attrs)

    if response is None:
        predictions_path = settings.MODELS_ROUTE + "/predictions.csv"
        print(
            "Could not connect to Orion Context Broker. Saving predictions in {}".format(
                predictions_path
            )
        )

        data = [date, entity_id]
        columns = ["Date", "Entity ID"]
        for key in results["models_predictions"].keys():
            data.append(results["models_predictions"][key])
            columns.append(key)

        to_append = pd.DataFrame([data], columns=columns)
        reader = CSVReader(path=predictions_path)
        reader.append_to_csv(to_append)

    return {"current": 100, "total": 100, "status": "TASK ENDED", "results": results}
