from unittest import TestCase
from keras.regularizers import l1
from blackbox.blackbox import BlackBoxAnomalyDetection, NotAnomalyModelClass
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
from blackbox.models.base_model import ModelNotTrained
from blackbox.utils.csv import CSVReader


class TestBlackBoxAnomalyDetection(TestCase):
    """Tests for Blackbox"""

    def setUp(self) -> None:
        self.model = BlackBoxAnomalyDetection(scaler="minmax", verbose=True)
        self.model.add_model(AnomalyPCAMahalanobis(verbose=True))
        self.model.add_model(
            AnomalyAutoencoder(verbose=True, hidden_neurons=[4, 2, 2, 4])
        )
        self.model.add_model(AnomalyKMeans(verbose=True))
        self.model.add_model(AnomalyOneClassSVM(verbose=True))
        self.model.add_model(AnomalyIsolationForest(verbose=True))
        self.model.add_model(AnomalyGaussianDistribution(verbose=True))
        self.model.add_model(AnomalyKNN(verbose=True))
        self.model.add_model(AnomalyLOF(verbose=True))

        self.model_name = BlackBoxAnomalyDetection(scaler="standard", verbose=True)
        self.model_name.add_model(AnomalyPCAMahalanobis(verbose=True), "PCAMahalanobis")
        self.model_name.add_model(
            AnomalyAutoencoder(
                kernel_regularizer=l1(0.0), verbose=True, hidden_neurons=[4, 2, 2, 4]
            ),
            "Autoencoder",
        )
        self.model_name.add_model(AnomalyKMeans(n_clusters=20, verbose=True), "KMeans")
        self.model_name.add_model(AnomalyOneClassSVM(verbose=True), "OneClassSVM")
        self.model_name.add_model(
            AnomalyIsolationForest(verbose=True), "IsolationForest"
        )
        self.model_name.add_model(
            AnomalyGaussianDistribution(verbose=True), "GaussianDistribution"
        )
        self.model_name.add_model(AnomalyKNN(verbose=True), "KNearestNeighbors")
        self.model_name.add_model(AnomalyLOF(verbose=True), "LocalOutlierFactor")

    def test_add_model(self):
        """Tests that models are correctly added to the Blackbox"""
        self.assertIsInstance(
            self.model.models["AnomalyPCAMahalanobis"], AnomalyPCAMahalanobis
        )
        self.assertIsInstance(
            self.model.models["AnomalyAutoencoder"], AnomalyAutoencoder
        )
        self.assertIsInstance(self.model.models["AnomalyKMeans"], AnomalyKMeans)
        self.assertIsInstance(
            self.model.models["AnomalyOneClassSVM"], AnomalyOneClassSVM
        )
        self.assertIsInstance(
            self.model.models["AnomalyIsolationForest"], AnomalyIsolationForest
        )
        self.assertIsInstance(
            self.model.models["AnomalyGaussianDistribution"],
            AnomalyGaussianDistribution,
        )
        self.assertIsInstance(
            self.model.models["AnomalyKNN"], AnomalyKNN,
        )
        self.assertIsInstance(
            self.model.models["AnomalyLOF"], AnomalyLOF,
        )

        self.assertIsInstance(
            self.model_name.models["PCAMahalanobis"], AnomalyPCAMahalanobis
        )
        self.assertIsInstance(self.model_name.models["Autoencoder"], AnomalyAutoencoder)
        self.assertIsInstance(self.model_name.models["KMeans"], AnomalyKMeans)
        self.assertIsInstance(self.model_name.models["OneClassSVM"], AnomalyOneClassSVM)
        self.assertIsInstance(
            self.model_name.models["IsolationForest"], AnomalyIsolationForest
        )
        self.assertIsInstance(
            self.model_name.models["GaussianDistribution"], AnomalyGaussianDistribution
        )
        self.assertIsInstance(self.model_name.models["KNearestNeighbors"], AnomalyKNN)
        self.assertIsInstance(self.model_name.models["LocalOutlierFactor"], AnomalyLOF)

    def test_add_model_no_class_anomaly(self):
        """Tests that an error is raised if an instance of a class that doesn't inherit from AnomalyClassModel
        is added to the blackbox"""
        bb = BlackBoxAnomalyDetection()
        self.assertRaises(NotAnomalyModelClass, bb.add_model, 1)

    def test_train_save_load_predict_model(self):
        """Tests training, saving, loading and predicting  a Blackbox"""
        reader = CSVReader("./test_csv/no_anomaly_data.csv")  # read csv
        df = reader.get_df()

        reader2 = CSVReader("./test_csv/all_data_anomalies_included.csv")
        df2 = reader2.get_df()

        print(df.shape, df2.shape)

        def cb_function(progress, message):
            print(message, "Progress: ", progress)

        self.assertRaises(ModelNotTrained, self.model.flag_anomaly, df2)

        self.model.train_models(df, cb_function)  # train model

        # save the whole Blackbox and one by one each model
        self.model.save_blackbox()
        self.model.save_models()

        # load the whole Blackbox and one by one each model
        self.model.load_blackbox()
        self.model.load_models()

        # predict
        results = self.model.flag_anomaly(df2)
        print(results)
