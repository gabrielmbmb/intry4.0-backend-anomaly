from unittest import TestCase
from blackbox.blackbox import BlackBoxAnomalyDetection
from blackbox.models import AnomalyPCAMahalanobis, AnomalyAutoencoder, AnomalyKMeans, AnomalyIsolationForest, \
    AnomalyGaussianDistribution, AnomalyOneClassSVM


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
