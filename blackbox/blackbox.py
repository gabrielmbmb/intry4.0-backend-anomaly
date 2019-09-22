import os
import pickle
import numpy as np

from blackbox.models import AnomalyModel
from sklearn.preprocessing import MinMaxScaler


class NotAnomalyModelClass(Exception):
    """Raised when a model added to the blackbox is not an instance of AnomalyModel"""
    pass


class BlackBoxAnomalyDetection:
    """
    Class in charge of reading the training data from a given source and executing the Anomaly Detections models.

    Args:
         verbose (bool): verbose mode. Defaults to False.

    Raises:
        NotAnomalyModelClass: when trying to add a model that is not an instance of AnomalyModel.
    """

    def __init__(self, verbose=False):
        self.models = {}
        self.verbose = verbose
        self.scaler = None

    def add_model(self, model, name=None) -> None:
        """
        Adds an Anomaly Detection Model to the blackbox.

        Args:
            model (AnomalyModel): Anomaly Detection Model.
            name (str): name of the model. Defaults to '<ClassName>'.
        """
        if not isinstance(model, AnomalyModel):
            raise NotAnomalyModelClass('The model to be added is not an instance of blackbox.models.AnomalyModel!')

        if name is None:
            name = model.__class__.__name__

        if self.verbose:
            print('Adding model {} to the blackbox...'.format(model.__class__.__name__))

        self.models[name] = model

    def scale_data(self, data) -> np.ndarray:
        """
        Scales the data before training or making a prediction with a Min Max Scaler which is sensitive to outliers. The
        first time that the function is called, the scaler will be fitted with the data passed. Then, the trained scaler
        will be used to scale the data.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data to be scaled.

        Returns:
            numpy.ndarray: scaled data.
        """
        if self.scaler is None:
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            self.scaler = scaler
            return scaled_data

        scaled_data = self.scaler.transform(data)
        return scaled_data

    def train_models(self, data, cb_func=None) -> None:
        """
        Trains the models in the blackbox.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data with no anomalies.
            cb_func (function): callback function that will be executed when the training starts for a model. Progress
                and a message will be passed to this function. Defaults to None.
        """
        data = self.scale_data(data)

        model_n = 0
        for name, model in self.models.items():
            if cb_func:
                progress = (model_n / len(self.models)) * 100
                message = 'Training model ' + name
                cb_func(progress, message)

            if self.verbose:
                print('Training model {}...'.format(name))

            model.train(data)
            model_n += 1

        if self.verbose:
            print('Models trained!')

    def flag_anomaly(self, data) -> np.ndarray:
        """
        Determines if a data point is an anomaly or not using the models in the blackbox.

        Args:
            data (numpy.ndarray or pandas.DataFrame or list): data.

        Returns:
            numpy.ndarray: list containing list of bool indicating if the data point is an anomaly.
        """
        if isinstance(data, list):
            data = np.array(data)

        results = []
        for name, model in self.models.items():
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

        for name, model in self.models.items():
            path = path_dir + '/' + name + '.pkl'
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
            path = path_dir + '/' + self.models[model].__class__.__name__ + '.pkl'
            if self.verbose:
                print('Loading model from {}'.format(path))
            self.models[model].load_model(path=path)

    def save_blackbox(self, path='./blackbox.pkl') -> str:
        """
        Saves the entire Blackbox, that's means that all models will be saved in the same file and it will be easier to
        load the Blackbox instead of loading every model one by one and then adding it to the Blackbox.

        Args:
            path (str): path to save the Blackbox. Defaults to './blackbox.pkl'

        Returns:
            str: path of the saved blackbox.
        """

        try:
            with open(path, 'wb') as f:
                pickle.dump(self.models, f)
        except pickle.PicklingError as e:
            print('PicklingError: ', str(e))
        except Exception as e:
            print('An error has occurred when trying to write the file: ', str(e))

        return path

    def load_blackbox(self, path='./blackbox.pkl') -> None:
        """
        Loads a Blackbox from a pickle file.

        Args:
            path (str): path from where to load the Blackbox. Defaults to './blackbox.pkl'.
        """
        loaded_data = None

        try:
            with open(path, 'rb') as f:
                loaded_data = pickle.load(f)
        except pickle.UnpicklingError as e:
            print('UnpicklingError: ', str(e))
        except Exception as e:
            print('An error has occurred when trying to write the file: ', str(e))

        self.models = loaded_data
