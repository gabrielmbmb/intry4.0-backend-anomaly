import pickle
from abc import ABCMeta, abstractmethod


class ModelNotTrained(Exception):
    """Raised when a model is not trained"""

    pass


class AnomalyModel(metaclass=ABCMeta):
    """Abstract base class for anomaly detection model."""

    TRAIN_PARAMS = []

    def __init__(self):
        self._verbose = None

    @abstractmethod
    def train(self, X, y):
        """This method should implement all the logic to train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """
        This method should implement all the logic to predict a value in order to 
        flag as anomaly.
        """
        pass

    @abstractmethod
    def flag_anomaly(self, X):
        """
        This method should implement a metric to flag a data point as anomaly or not
        anomalous.
        """
        pass

    def save_model(self, path=None) -> str:
        """
        Saves the trained model in a Pickle file. Only Class Attributes that start with
        '_' will be saved.

        Args:
            path (str): path to save the model. Defaults to './<Class Name>.pkl'

        Returns:
            str: path of the saved file.
        """
        if path is None:
            path = "./" + self.__class__.__name__ + ".pkl"

        data_to_pickle = {}

        for key, value in self.__dict__.items():
            # save only protected attributes
            if key[0] == "_":
                data_to_pickle[key] = value

        if self._verbose:
            print("Saving model to {}".format(path))

        try:
            with open(path, "wb") as f:
                pickle.dump(data_to_pickle, f)
        except pickle.PicklingError as e:
            print("PicklingError: ", str(e))
        except Exception as e:
            print("An error has occurred when trying to write the file: ", str(e))

        return path

    def load_model(self, path=None) -> None:
        """
        Load a trained model from a Pickle file.

        Args:
            path (str): path from where to load the model. Defaults to
                './<Class Name>.pkl'
        """
        loaded_data = None

        if path is None:
            path = "./" + self.__class__.__name__ + ".pkl"

        if self._verbose:
            print("Loading model from {}".format(path))

        try:
            with open(path, "rb") as f:
                loaded_data = pickle.load(f)
        except pickle.UnpicklingError as e:
            print("UnpicklingError: ", str(e))
        except Exception as e:
            print("An error has occurred when trying to read the file: ", str(e))

        if loaded_data:
            for key, value in loaded_data.items():
                self.__setattr__(key, value)

    def check_if_trained(self):
        """
        Check if the model is already trained.

        Returns:
            bool: indicating whether the model is trained or not.
        """

        for param in self.TRAIN_PARAMS:
            if self.__dict__.get(param) is None:
                return False

        return True
