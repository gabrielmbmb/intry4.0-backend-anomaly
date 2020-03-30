import copy
import pytest
import pandas as pd
from keras.regularizers import l1
from blackbox.blackbox import BlackBoxAnomalyDetection
from blackbox.models.base_model import ModelNotTrained
from blackbox.models.unsupervised import (
    AnomalyPCAMahalanobis,
    AnomalyAutoencoder,
    AnomalyKMeans,
    AnomalyIsolationForest,
    AnomalyGaussianDistribution,
    AnomalyOneClassSVM,
    AnomalyKNN,
    AnomalyLOF,
)


class TestBlackBoxAnomalyDetection:
    """Tests for Blackbox"""

    @pytest.fixture(scope="class", autouse=True)
    def setup(self) -> None:
        cls = type(self)
        cls.model = BlackBoxAnomalyDetection(scaler="minmax", verbose=True)
        cls.train_df = pd.read_csv("./tests/test_csv/no_anomaly_data.csv", index_col=0)
        cls.predict_df = pd.read_csv(
            "./tests/test_csv/all_data_anomalies_included.csv", index_col=0
        )

    def test_add_model(self):
        """Tests that models are correctly added to the Blackbox."""
        self.model.add_model("pca_mahalanobis")
        self.model.add_model(
            "autoencoder",
            **{"hidden_neurons": [4, 2, 2, 4], "kernel_regularizer": l1(0.0)},
        )
        self.model.add_model("kmeans")
        self.model.add_model("one_class_svm")
        self.model.add_model("gaussian_distribution")
        self.model.add_model("isolation_forest")
        self.model.add_model("knearest_neighbors")
        self.model.add_model("local_outlier_factor")

        assert isinstance(self.model.models["pca_mahalanobis"], AnomalyPCAMahalanobis)
        assert isinstance(self.model.models["autoencoder"], AnomalyAutoencoder)
        assert isinstance(self.model.models["kmeans"], AnomalyKMeans)
        assert isinstance(self.model.models["one_class_svm"], AnomalyOneClassSVM)
        assert isinstance(self.model.models["isolation_forest"], AnomalyIsolationForest)
        assert isinstance(
            self.model.models["gaussian_distribution"], AnomalyGaussianDistribution,
        )
        assert isinstance(self.model.models["knearest_neighbors"], AnomalyKNN,)
        assert isinstance(self.model.models["local_outlier_factor"], AnomalyLOF,)

        with pytest.raises(KeyError):
            self.model.add_model("foo")

    def test_model_not_trained(self):
        """Tests that ModelNotTrained is raised."""
        with pytest.raises(ModelNotTrained):
            self.model.flag_anomaly(self.predict_df)

    def test_train_model(self):
        """Tests training a Blackbox model."""

        def cb_function(progress, message):
            print(message, "Progress: ", progress)

        copy_model = copy.copy(self.model)
        copy_model.scaler = "standard"
        copy_model.train_models(self.train_df, y=None, cb_func=cb_function)
        self.model.train_models(self.train_df, y=None, cb_func=cb_function)

    def test_predict_model(self):
        """Tests predicting with a Blackbox model"""
        results = self.model.flag_anomaly(self.predict_df)

    def test_save_load_model(self):
        """Tests saving and loading a Blackbox model."""
        pickled = self.model.save()
        assert isinstance(pickled, bytes)

        new_model = BlackBoxAnomalyDetection()
        new_model.load(pickled)

        assert all(
            [
                type(model1) == type(model2)
                for model1, model2 in zip(self.model.models, new_model.models)
            ]
        )
        assert type(self.model.scaler) == type(new_model.scaler)
        assert self.model.verbose == new_model.verbose
