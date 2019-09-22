from unittest import TestCase
from blackbox.blackbox import BlackBoxAnomalyDetection
from blackbox.models import AnomalyPCAMahalanobis, AnomalyAutoencoder, AnomalyKMeans, AnomalyIsolationForest, \
    AnomalyGaussianDistribution, AnomalyOneClassSVM
from blackbox.csv_reader import CSVReader


class TestBlackBoxAnomalyDetection(TestCase):
    """Tests for Blackbox"""

    def setUp(self) -> None:
        self.model = BlackBoxAnomalyDetection()
        self.model.add_model(AnomalyPCAMahalanobis())
        self.model.add_model(AnomalyAutoencoder())
        self.model.add_model(AnomalyKMeans())
        self.model.add_model(AnomalyOneClassSVM())
        self.model.add_model(AnomalyIsolationForest())
        self.model.add_model(AnomalyGaussianDistribution())

        self.model_name = BlackBoxAnomalyDetection()
        self.model_name.add_model(AnomalyPCAMahalanobis(), 'PCAMahalanobis')
        self.model_name.add_model(AnomalyAutoencoder(), 'Autoencoder')
        self.model_name.add_model(AnomalyKMeans(), 'KMeans')
        self.model_name.add_model(AnomalyOneClassSVM(), 'OneClassSVM')
        self.model_name.add_model(AnomalyIsolationForest(), 'IsolationForest')
        self.model_name.add_model(AnomalyGaussianDistribution(), 'GaussianDistribution')

    def test_add_model(self):
        """Tests that models are correctly added to the Blackbox"""
        self.assertIsInstance(self.model.models['AnomalyPCAMahalanobis'], AnomalyPCAMahalanobis)
        self.assertIsInstance(self.model.models['AnomalyAutoencoder'], AnomalyAutoencoder)
        self.assertIsInstance(self.model.models['AnomalyKMeans'], AnomalyKMeans)
        self.assertIsInstance(self.model.models['AnomalyOneClassSVM'], AnomalyOneClassSVM)
        self.assertIsInstance(self.model.models['AnomalyIsolationForest'], AnomalyIsolationForest)
        self.assertIsInstance(self.model.models['AnomalyGaussianDistribution'], AnomalyGaussianDistribution)

        self.assertIsInstance(self.model_name.models['PCAMahalanobis'], AnomalyPCAMahalanobis)
        self.assertIsInstance(self.model_name.models['Autoencoder'], AnomalyAutoencoder)
        self.assertIsInstance(self.model_name.models['KMeans'], AnomalyKMeans)
        self.assertIsInstance(self.model_name.models['OneClassSVM'], AnomalyOneClassSVM)
        self.assertIsInstance(self.model_name.models['IsolationForest'], AnomalyIsolationForest)
        self.assertIsInstance(self.model_name.models['GaussianDistribution'], AnomalyGaussianDistribution)

    def test_train_save_load_predict_model(self):
        """Tests training, saving, loading and predicting  a Blackbox"""
        reader = CSVReader('./tests/train_data.csv')  # read csv
        df = reader.get_df()

        def cb_function(progress, message):
            print(message, 'Progress: ', progress)

        self.model.train_models(df, cb_function)  # train model

        # save the whole Blackbox and one by one each model
        self.model.save_blackbox()
        self.model.save_models()

        # load the whole Blackbox and one by one each model
        self.model.load_blackbox()
        self.model.load_models()

        # predict
        results = self.model.flag_anomaly([[0.72, 0.84, 0.22, 0.66]])
