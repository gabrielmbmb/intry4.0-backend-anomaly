from unittest import TestCase
from blackbox.blackbox import BlackBoxAnomalyDetection, NotAnomalyModelClass
from blackbox.models import AnomalyPCAMahalanobis, AnomalyAutoencoder, AnomalyKMeans, AnomalyIsolationForest, \
    AnomalyGaussianDistribution, AnomalyOneClassSVM
from blackbox.utils.csv import CSVReader
from keras.regularizers import l1

class TestBlackBoxAnomalyDetection(TestCase):
    """Tests for Blackbox"""

    def setUp(self) -> None:
        self.model = BlackBoxAnomalyDetection(verbose=True)
        self.model.add_model(AnomalyPCAMahalanobis(verbose=True))
        self.model.add_model(AnomalyAutoencoder(verbose=True))
        self.model.add_model(AnomalyKMeans(verbose=True))
        self.model.add_model(AnomalyOneClassSVM(verbose=True))
        self.model.add_model(AnomalyIsolationForest(verbose=True))
        self.model.add_model(AnomalyGaussianDistribution(verbose=True))

        self.model_name = BlackBoxAnomalyDetection(verbose=True)
        self.model_name.add_model(AnomalyPCAMahalanobis(verbose=True), 'PCAMahalanobis')
        self.model_name.add_model(AnomalyAutoencoder(kernel_regularizer=l1(0.0), verbose=True), 'Autoencoder')
        self.model_name.add_model(AnomalyKMeans(_n_clusters=20, verbose=True), 'KMeans')
        self.model_name.add_model(AnomalyOneClassSVM(verbose=True), 'OneClassSVM')
        self.model_name.add_model(AnomalyIsolationForest(verbose=True), 'IsolationForest')
        self.model_name.add_model(AnomalyGaussianDistribution(verbose=True), 'GaussianDistribution')

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

    def test_add_model_no_class_anomaly(self):
        """Tests that an error is raised if an instance of a class that doesn't inherit from AnomalyClassModel
        is added to the blackbox"""
        bb = BlackBoxAnomalyDetection()
        self.assertRaises(NotAnomalyModelClass, bb.add_model, 1)

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
