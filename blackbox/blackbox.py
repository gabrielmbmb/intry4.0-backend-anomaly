import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
from blackbox.available_models import AVAILABLE_MODELS


class BlackBoxAnomalyDetection:
    """
    Class in charge of reading the training data from a given source and executing the
        Anomaly Detections models.

    Args:
        scaler (str): scaler that will be to scale the data before training. Available
            scalers are 'minmax' and 'standard'. Defaults to 'minmax.
        verbose (bool): verbose mode. Defaults to False.

    Raises:
        KeyError: when trying to add a model that does not exist.
    """

    MODELS_CLASS = {
        "pca_mahalanobis": AnomalyPCAMahalanobis,
        "autoencoder": AnomalyAutoencoder,
        "kmeans": AnomalyKMeans,
        "one_class_svm": AnomalyOneClassSVM,
        "gaussian_distribution": AnomalyGaussianDistribution,
        "isolation_forest": AnomalyIsolationForest,
        "knearest_neighbors": AnomalyKNN,
        "local_outlier_factor": AnomalyLOF,
    }

    def __init__(self, scaler="minmax", verbose=False):
        self.models = {}
        self.scaler = scaler
        self.verbose = verbose
        self.scaler_model = None

    def add_model(self, model, **kwargs) -> None:
        """
        Adds an Anomaly Detection Model to the blackbox.

        Args:
            model (str): Anomaly Detection model.
            **kwargs: Parameters of Anomaly Detection model.

        Raises:
            KeyError: if model provided does not exist.
        """
        try:
            model_to_add = self.MODELS_CLASS[model](verbose=self.verbose, **kwargs)
        except KeyError:
            raise KeyError(
                f"Model {model} does not exist. Available models: "
                f"{', '.join(AVAILABLE_MODELS)}"
            )

        if self.verbose:
            print(f"Adding model {model} to the Blackbox...")

        self.models[model] = model_to_add

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

        results = {}
        for model_name, model in self.models.items():
            results[model_name] = model.flag_anomaly(X).tolist()

        return results

    def save(self) -> bytes:
        """
        Generates a pickle file of the Blackbox model.

        Returns:
            bytes: pickled Blackbox.
        """
        data_to_pickle = {
            "models": self.models,
            "scaler": self.scaler,
            "verbose": self.verbose,
        }

        try:
            pickled_blackbox = pickle.dumps(data_to_pickle)
        except pickle.PicklingError as e:
            print("PicklingError: ", str(e))
            return None

        return pickled_blackbox

    def load(self, pickled_blackbox) -> None:
        """
        Loads a Blackbox from a pickle.

        Args:
            pickled_blackbox (bytes): pickle with Blackbox to load.

        Raises:
            KeyError: if pickle does not contains attribute key.
        """
        loaded_data = None

        try:
            loaded_data = pickle.loads(pickled_blackbox)
        except pickle.UnpicklingError as e:
            loaded_data = None
            print("UnpicklingError: ", str(e))

        if loaded_data:
            try:
                self.models = loaded_data["models"]
                self.scaler = loaded_data["scaler"]
                self.verbose = loaded_data["verbose"]
            except KeyError:
                print("Keys not found in pickle!")
