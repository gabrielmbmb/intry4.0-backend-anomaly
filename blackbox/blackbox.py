import os
import numpy as np

from .models import AnomalyAutoencoder, AnomalyOneClassSVM, AnomalyGaussianDistribution, AnomalyIsolationForest, \
    AnomalyKMeans, AnomalyPCAMahalanobis, AnomalyModel


class NotAnomalyModelClass(Exception):
    """Raised when a model added to the blackbox is not an instance of AnomalyModel"""
    pass


class BlackBoxAnomalyDetection:
    """
    Class in charge of reading the training data from a given source and executing the Anomaly Detections models.

    Args:
         verbose (bool): verbose mode. Defaults to False.
    """

    def __init__(self, verbose=False):
        self.models = []
        self.verbose = verbose

    def add_model(self, model) -> None:
        """
        Adds an Anomaly Detection Model to the blackbox.

        Args:
            model (AnomalyModel):  Anomaly Detection Model.
        """
        if not isinstance(model, AnomalyModel):
            raise NotAnomalyModelClass('The model to be added is not an instance of blackbox.models.AnomalyModel!')

        if self.verbose:
            print('Adding model {} to the blackbox...'.format(model.__class__.__name__))

        self.models.append(model)

    def train_models(self, data) -> None:
        """
        Trains the models in the blackbox.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data with no anomalies.
        """
        for model in self.models:
            if self.verbose:
                print('Training model {}...'.format(model.__class__.__name__))
            model.train(data)

    def flag_anomaly(self, data) -> np.ndarray:
        """
        Determines if a data point is an anomaly or not using the models in the blackbox.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data.

        Returns:
            numpy.ndarray: list containing list of bool indicating if the data point is an anomaly.
        """
        results = []
        for model in self.models:
            results.append(model.flag_anomaly(data))

        np_array = np.array(results)
        np_array = np_array.reshape((np_array.shape[1], np_array.shape[0]))

        return np_array

    def save_models(self, path_dir='./saved_models') -> None:
        """
        Saves the trained models in the directory specified.

        Args:
            path_dir (str): path to the directory where to save the files. Defaults to './saved_models'.
        """
        if not os.path.exists(path_dir):
            if self.verbose:
                print('Directory {} does not exists. Creating...'.format(path_dir))
            os.mkdir(path_dir)

        for model in self.models:
            path = path_dir + '/' + model.__class__.__name__ + '.pkl'
            if self.verbose:
                print('Saving model in {}'.format(path))
            model.save_model(path=path)

    def load_models(self, path_dir='./saved_models') -> None:
        """
        Loads the trained models from the directory specified.

        Args:
            path_dir (str): path to the directory storing the saved models. Defaults to './saved_models'.
        """
        for model in self.models:
            path = path_dir + '/' + model.__class__.__name__ + '.pkl'
            if self.verbose:
                print('Loading model from {}'.format(path))
            model.load_model(path=path)