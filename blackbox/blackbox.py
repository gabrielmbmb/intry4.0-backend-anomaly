import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from blackbox.models.base_model import AnomalyModel

AVAILABLE_MODELS = [
    "PCAMahalanobis",
    "Autoencoder",
    "KMeans",
    "OneClassSVM",
    "GaussianDistribution",
    "IsolationForest",
    "KNearestNeighbors",
    "LocalOutlierFactor",
]


class NotAnomalyModelClass(Exception):
    """Raised when a model added to the blackbox is not an instance of AnomalyModel"""


class BlackBoxAnomalyDetection:
    """
    Class in charge of reading the training data from a given source and executing the
        Anomaly Detections models.

    Args:
        scaler (str): scaler that will be to scale the data before training. Available
            scalers are 'minmax' and 'standard'. Defaults to 'minmax.
        verbose (bool): verbose mode. Defaults to False.

    Raises:
        NotAnomalyModelClass: when trying to add a model that is not an instance of
            AnomalyModel.
    """

    def __init__(self, scaler="minmax", verbose=False):
        self.models = {}
        self.scaler = scaler
        self.verbose = verbose
        self.scaler_model = None

    def add_model(self, model, name=None) -> None:
        """
        Adds an Anomaly Detection Model to the blackbox.

        Args:
            model (AnomalyModel): Anomaly Detection Model.
            name (str): name of the model. Defaults to '<ClassName>'.
        """
        if not isinstance(model, AnomalyModel):
            raise NotAnomalyModelClass(
                "The model to be added is not an instance of blackbox.models.AnomalyModel!"
            )

        if name is None:
            name = model.__class__.__name__

        if self.verbose:
            print("Adding model {} to the blackbox...".format(model.__class__.__name__))

        self.models[name] = model

    def scale_data(self, X) -> np.ndarray:
        """
        Scales the data before training or making a prediction with a Min Max Scaler
        which is sensitive to outliers. The first time that the function is called, the
        scaler will be fitted with the data passed. Then, the trained scaler will be
        used to scale the data.

        Args:
            X (numpy.ndarray or pandas.DataFrame): data to be scaled.

        Returns:
            numpy.ndarray: scaled data.
        """
        if self.scaler == "minmax" and self.scaler_model is None:
            if self.verbose:
                print("Fitting Min Max scaler...")

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(X)
            self.scaler = scaler
            return scaled_data

        elif self.scaler == "standard" and self.scaler_model is None:
            if self.verbose:
                print("Fitting Standard scaler")

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(X)
            self.scaler = scaler
            return scaled_data

        if self.verbose:
            print("Scaling data...")

        scaled_data = self.scaler.transform(X)
        return scaled_data

    def train_models(self, X, y=None, cb_func=None) -> None:
        """
        Trains the models in the blackbox.

        Args:
            X (numpy.ndarray or pandas.DataFrame): training data.
            y (numpy.ndarray or pandas.DataFrame): training labels which will be only
                used with the supervised models.
            cb_func (function): callback function that will be executed when the
                training starts for a model. Progress and a message will be passed to
                this function. Defaults to None.
        """
        X = self.scale_data(X)

        model_n = 0
        for name, model in self.models.items():
            if cb_func:
                progress = (model_n / len(self.models)) * 100
                message = "Training model " + name
                cb_func(progress, message)

            if self.verbose:
                print("Training model {}...".format(name))

            model.train(X, y)
            model_n += 1

        if self.verbose:
            print("Models trained!")

    def flag_anomaly(self, X) -> np.ndarray:
        """
        Determines if a data point is an anomaly or not using the models in the blackbox.

        Args:
            X (numpy.ndarray or pandas.DataFrame or list): data.

        Returns:
            numpy.ndarray: list containing list of bool indicating if the data point is
                an anomaly.
        """
        if isinstance(X, list):
            X = np.array(X)

        X = self.scale_data(X)

        results = []
        for _, model in self.models.items():
            results.append(model.flag_anomaly(X))

        np_array = np.array(results)
        return np_array

    def save_models(self, path_dir="./saved_models") -> None:
        """
        Saves the trained models in the directory specified.

        Args:
            path_dir (str): path to the directory where to save the files. Defaults to
                './saved_models'.
        """
        if not os.path.exists(path_dir):
            if self.verbose:
                print("Directory {} does not exists. Creating...".format(path_dir))
            os.mkdir(path_dir)

        for name, model in self.models.items():
            path = path_dir + "/" + name + ".pkl"
            if self.verbose:
                print("Saving model in {}".format(path))
            model.save_model(path=path)

    def load_models(self, path_dir="./saved_models") -> None:
        """
        Loads the trained models from the directory specified.

        Args:
            path_dir (str): path to the directory storing the saved models. Defaults to
                './saved_models'.
        """
        for model in self.models:
            path = path_dir + "/" + self.models[model].__class__.__name__ + ".pkl"
            if self.verbose:
                print("Loading model from {}".format(path))
            self.models[model].load_model(path=path)

    def save_blackbox(self, path="./blackbox.pkl") -> str:
        """
        Saves the entire Blackbox, that's means that all models will be saved in the
        same file and it will be easier to load the Blackbox instead of loading every
        model one by one and then adding it to the Blackbox.

        Args:
            path (str): path to save the Blackbox. Defaults to './blackbox.pkl'

        Returns:
            str: path of the saved blackbox.
        """
        data_to_pickle = {"models": self.models, "scaler": self.scaler}

        try:
            with open(path, "wb") as f:
                pickle.dump(data_to_pickle, f)
        except pickle.PicklingError as e:
            print("PicklingError: ", str(e))
        except Exception as e:
            print("An error has occurred when trying to write the file: ", str(e))

        return path

    def load_blackbox(self, path="./blackbox.pkl") -> None:
        """
        Loads a Blackbox from a pickle file.

        Args:
            path (str): path from where to load the Blackbox. Defaults to
                './blackbox.pkl'.
        """
        loaded_data = None

        try:
            with open(path, "rb") as f:
                loaded_data = pickle.load(f)
        except pickle.UnpicklingError as e:
            print("UnpicklingError: ", str(e))
        except Exception as e:
            print("An error has occurred when trying to write the file: ", str(e))

        self.models = loaded_data["models"]
        self.scaler = loaded_data["scaler"]
