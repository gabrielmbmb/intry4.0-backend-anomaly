import numpy as np
from settings import MODELS_ROUTE, MODELS_ROUTE_JSON
from celery import Celery
from settings import APP_NAME, CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from blackbox.api.api_utils import add_model_entity_json
from blackbox.blackbox import BlackBoxAnomalyDetection
from blackbox.models import AnomalyPCAMahalanobis, AnomalyAutoencoder, AnomalyKMeans, AnomalyIsolationForest, \
    AnomalyGaussianDistribution, AnomalyOneClassSVM
from blackbox.csv_reader import CSVReader
import pandas as pd

celery_app = Celery(APP_NAME, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

# Tasks
@celery_app.task(bind=True)
def train_blackbox(self, entity_id, filename, model_name):
    """
    Celery's task which will read the training data and train the Blackbox model for the specified entity.

    Args:
        self (object): Celery app object.
        entity_id (str): Orion Context Broker (FIWARE component) entity ID
        filename (str): path of the file which will be used for training the model.
        model_name (str): model name which will be used to refer to the Blackbox trained model.

    Returns:
        dict: status of the task.
    """
    def cb_function(progress, message):
        self.update_state(state='PROGRESS', meta={'current': progress, 'total': 100, 'status': message})

    # read CSV file
    # reader = CSVReader(filename)
    # data = reader.get_df()
    data = pd.read_csv(filename, index_col=0)

    # create and train Blackbox model
    model = BlackBoxAnomalyDetection(verbose=True)
    model.add_model(AnomalyPCAMahalanobis())
    model.add_model(AnomalyAutoencoder())
    model.add_model(AnomalyKMeans())
    model.add_model(AnomalyOneClassSVM())
    model.add_model(AnomalyIsolationForest())
    model.add_model(AnomalyGaussianDistribution())
    model.train_models(data, cb_func=cb_function)

    # save the model
    model_path = MODELS_ROUTE + '/' + entity_id + '/' + model_name + '.pkl'
    model.save_blackbox(model_path)

    # add the model to the JSON
    add_model_entity_json(MODELS_ROUTE_JSON, entity_id, model_name, model_path, filename)

    return {'current': 100, 'total': 100, 'status': 'TASK ENDED'}


@celery_app.task()
def predict_blackbox(entity_id, model_path, predict_data):
    model = BlackBoxAnomalyDetection(verbose=True)
    model.load_blackbox(model_path)
    predict_data = np.array([predict_data])
    predictions = model.flag_anomaly(predict_data)
    print(predictions)
    return {'current': 100, 'total': 100, 'status': 'TASK ENDED', 'results': predictions}
