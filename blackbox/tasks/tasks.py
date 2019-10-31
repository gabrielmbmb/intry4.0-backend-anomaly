import numpy as np
from blackbox import settings
from celery import Celery
from blackbox.api.utils import add_model_entity_json
from blackbox.blackbox import BlackBoxAnomalyDetection
from blackbox.models import AnomalyPCAMahalanobis, AnomalyAutoencoder, AnomalyKMeans, AnomalyIsolationForest, \
    AnomalyGaussianDistribution, AnomalyOneClassSVM
from blackbox.csv_reader import CSVReader

# Todo: add logging to tasks
# Todo: send the predictions results to somewhere (not defined yet)

celery_app = Celery('tasks',
                    broker=settings.CELERY_BROKER_URL,
                    backend=settings.CELERY_RESULT_BACKEND)


# Tasks
@celery_app.task(name='tasks.train', bind=True)
def train_blackbox(self, entity_id, filename, model_name, models, input_arguments):
    """
    Celery's task which will read the training data and train the Blackbox model for the specified entity.

    Args:
        self (object): Celery app object.
        entity_id (str): Orion Context Broker (FIWARE component) entity ID
        filename (str): path of the file which will be used for training the model.
        model_name (str): model name which will be used to refer to the Blackbox trained model.
        models (list of str): list of strings containing the name of anomaly prediction models
            that are going to be used.
        input_arguments (list of str): name of the inputs variables of the Blackbox model.

    Returns:
        dict: status of the task.
    """

    def cb_function(progress, message):
        self.update_state(state='PROGRESS', meta={'current': progress, 'total': 100, 'status': message})

    # read CSV file
    reader = CSVReader(filename)
    data = reader.get_df()

    # create and train Blackbox model
    model = BlackBoxAnomalyDetection(verbose=True)

    if BlackBoxAnomalyDetection.AVAILABLE_MODELS[0] in models:
        model.add_model(AnomalyPCAMahalanobis())

    if BlackBoxAnomalyDetection.AVAILABLE_MODELS[1] in models:
        model.add_model(AnomalyAutoencoder())

    if BlackBoxAnomalyDetection.AVAILABLE_MODELS[2] in models:
        model.add_model(AnomalyKMeans())

    if BlackBoxAnomalyDetection.AVAILABLE_MODELS[3] in models:
        model.add_model(AnomalyOneClassSVM())

    if BlackBoxAnomalyDetection.AVAILABLE_MODELS[4] in models:
        model.add_model(AnomalyIsolationForest())

    if BlackBoxAnomalyDetection.AVAILABLE_MODELS[5] in models:
        model.add_model(AnomalyGaussianDistribution())

    model.train_models(data, cb_func=cb_function)

    # save the model
    model_path = settings.MODELS_ROUTE + '/' + entity_id + '/' + model_name + '.pkl'
    model.save_blackbox(model_path)

    # add the model to the JSON
    add_model_entity_json(settings.MODELS_ROUTE_JSON, entity_id, model_name, model_path, filename, models,
                          input_arguments)

    return {'current': 100, 'total': 100, 'status': 'TASK ENDED'}


@celery_app.task(name='tasks.predict')
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
    results = {'entity_id': entity_id, 'date': date, 'models_predictions': {}}

    for n_model, model in enumerate(model.models.items()):
        results['models_predictions'][model[0]] = predictions[n_model]

    return {'current': 100, 'total': 100, 'status': 'TASK ENDED', 'results': results}
