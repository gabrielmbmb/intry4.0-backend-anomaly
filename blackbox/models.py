import numpy as np
import pandas as pd
import pickle

from abc import ABCMeta, abstractmethod
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from typing import List, Tuple


class AnomalyModel(metaclass=ABCMeta):
    """Abstract base class for anomaly detection model."""

    def __init__(self):
        self.verbose = None

    @abstractmethod
    def train(self, train):
        """This method should implement all the logic to train the model."""
        pass

    @abstractmethod
    def predict(self, data):
        """This method should implement all the logic to predict a value in order to flag as anomaly."""
        pass

    @abstractmethod
    def flag_anomaly(self, data):
        """This method should implement a metric to flag a data point as anomaly or not anomalous."""
        pass

    def save_model(self, path=None) -> None:
        """
        Saves the trained model in a Pickle file. Only Class Attributes that start with '_' will be saved.

        Args:
            path (str): path to save the model. Defaults to './<Class Name>.pkl'
        """
        if path is None:
            path = './' + self.__class__.__name__ + '.pkl'

        data_to_pickle = {}

        for key, value in self.__dict__.items():
            # save only protected attributes
            if key[0] == '_':
                data_to_pickle[key] = value

        if self.verbose:
            print('Saving model to {}'.format(path))

        try:
            pickle.dump(data_to_pickle, open(path, 'wb'))
        except pickle.PicklingError as e:
            print('PicklingError: ', str(e))
        except Exception as e:
            print('An error has occurred when trying to write the file: ', str(e))

    def load_model(self, path=None) -> None:
        """
        Load a trained model from a Pickle file.

        Args:
            path (str): path from where to load the model. Defaults to './<Class Name>.pkl'
        """
        loaded_data = None

        if path is None:
            path = './' + self.__class__.__name__ + '.pkl'

        if self.verbose:
            print('Loading model from {}'.format(path))

        try:
            loaded_data = pickle.load(open(path, 'rb'))
        except pickle.UnpicklingError as e:
            print('UnpicklingError: ', str(e))
        except Exception as e:
            print('An error has occurred when trying to read the file: ', str(e))

        if loaded_data:
            for key, value in loaded_data.items():
                self.__setattr__(key, value)


class AnomalyPCAMahalanobis(AnomalyModel):
    """
    Unsupervised anomaly detection model based on Primary Component Analysis and Mahalanobis Distance. The model is
    trained with data that is considered to not have anomalies (normal operation) and reduces the dimensionality
    of the data with PCA, which is a technique very sensible to outliers. When the model receives a new data point, it
    will reduce the dimensionality of the point with PCA and then will calculate the distance of this point to the
    train distribution. If the distance surpass a threshold previously established, then the data point is flagged as
    an anomaly.

    Args:
        n_components (int or float): number of components to which the data have to be reduced. Defaults to 2.
        std_deviation_num (int): number of the standard deviation used to establish the threshold. Defaults to 3.
        verbose (bool): verbose mode. Defaults to False.
    """
    def __init__(self, n_components=2, std_deviation_num=3, verbose=False) -> None:
        super().__init__()
        self._pca = PCA(n_components=n_components, svd_solver='full')
        self._data = None
        self._distances = None
        self._cov = None
        self._threshold = None
        self._std_dev_num = std_deviation_num
        self.verbose = verbose

    def train(self, data) -> None:
        """
        Trains the model with the train data given.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data
        """
        self._data = self._pca.fit_transform(data)
        self._distances = self.mahalanobis_distance(self._data)
        self.calculate_threshold()

    def predict(self, data) -> np.ndarray:
        """
        Calculates the Mahalanobis distance from the data given to the training data.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data

        Returns:
            numpy.ndarray: distances between the data given and training data.
        """
        data_pca = self._pca.transform(data)
        data_distance = self.mahalanobis_distance(data_pca)
        return data_distance

    def flag_anomaly(self, data) -> np.ndarray:
        """
        Flag the data points as anomaly if the calculated Mahalanobis distance surpass a established threshold.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data to flag as anomalous or not anomalous.

        Returns:
            numpy.ndarray: list of booleans telling if point is anomalous or not.
        """
        distances = self.predict(data)
        return distances > self._threshold

    def calculate_threshold(self, k=3) -> None:
        """
        Computes the threshold that has to surpass a distance of a point to be flagged as an anomaly.

        Args:
            k (int): standard deviations. Defaults to 3.

        Returns:
            float: threshold value
        """
        self._threshold = np.mean(self._distances) * k

    def mahalanobis_distance(self, x) -> np.ndarray:
        """
        Computes the Mahalanobis distance between each row of x and the data distribution.

        Args:
            x (numpy.ndarray or pandas.DataFrame): vector or matrix of data with p columns.

        Returns:
            numpy.ndarray: distance between each row of x and the data distribution.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values

        if self._cov is None:
            self._cov = np.cov(self._data, rowvar=False)
        inv_cov_mat = np.linalg.inv(self._cov)
        diff = x - np.mean(self._data)
        return np.sqrt(np.diagonal(np.dot(np.dot(diff, inv_cov_mat), diff.T)))


class AnomalyAutoencoder(AnomalyModel):

    def __init__(self, verbose=False):
        super().__init__()

    def train(self, data) -> None:
        pass

    def predict(self, data) -> None:
        pass

    def flag_anomaly(self):
        pass


class AnomalyKMeans(AnomalyModel):
    """
    Unsupervised anomaly detection model based on K-Means Clustering. The model is trained with data that doesn't
    contains anomalies. This will create 'k' similar clusters of data points. When new data points are received, we will
    predict the nearest cluster in which the data belongs, and then calculate the distance from the new data points to
    the cluster centroid. If this distance surpass a established threshold, then the data point will be flagged as an
    anomaly.

    Args:
          verbose (bool): verbose mode. Defaults to False.
    """
    def __init__(self, verbose=False):
        super().__init__()
        self.verbose = verbose
        self._kmeans = None
        self._n_clusters = None
        self._threshold = None
        self._distances = None

    def train(self, data) -> None:
        """
        Trains the model with the train data given. First, the optimal number of clusters is calculated with the Elbow
        Method. Then, a K-Means Cluster model is fitted with the optimal number of clusters.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data
        """
        if self.verbose:
            print('Calculating optimal number of clusters with Elbow Method...')

        (self._n_clusters, _) = self.elbow(data)

        if self.verbose:
            print('Optimal number of n_clusters: {}. Fitting model...'.format(self._n_clusters))

        self._kmeans = KMeans(n_clusters=self._n_clusters)
        self._kmeans.fit(data)
        self._distances = self.get_distance_by_point(data, self._kmeans.cluster_centers_, self._kmeans.labels_)

        if self.verbose:
            print('Calculating threshold value to flag an anomaly...')

        self._threshold = self.calculate_threshold()

    def predict(self, data) -> List[float]:
        """
        Calculates the cluster of each data point and then calculates the distance between the data point and the
        cluster centroid.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data

        Returns:
            list of float: distances between the data given and its assigned cluster centroid.
        """
        data_labels = self._kmeans.predict(data)
        distances = self.get_distance_by_point(data, self._kmeans.cluster_centers_, data_labels)
        return distances

    def flag_anomaly(self, data) -> np.ndarray:
        """
        Flag the data points as anomaly if the calculated distance between the point and its assigned cluster centroid
        surpass a established threshold.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data to flag as anomalous or not anomalous.

        Returns:
            numpy.ndarray: list of booleans telling if point is anomalous or not.
        """
        distances = self.predict(data)
        return distances > self._threshold

    def calculate_threshold(self) -> float:
        """
        Calculates the threshold that has to surpass a distance between a data point and its cluster to be flagged as
        an anomaly.

        Returns:
            float: threshold value
        """
        return max(self._distances)

    def elbow(self, data, max_clusters=100) -> tuple:
        """
        Takes training data and computes K-Means model for each number of n_clusters given and get the score of each
        K-Means model. Then, calculate the distance between the line that goes from the point of the first score to the
        point of the last score with each score of the K-Means models.

        Args:
            data (pandas.DataFrame): training data.
            max_clusters (int): number of maximum clusters.

        Returns:
            tuple: tuple containing:
                n_cluster (float): optimal number of clusters (Elbow point)
                scores (list): list of float with scores for every K-Means model.
        """
        if self.verbose:
            print('Computing K-Means models...')

        n_clusters = range(1, max_clusters)
        kmeans = [KMeans(n_clusters=i).fit(data) for i in n_clusters]
        scores = [kmeans[i].score(data) for i in range(len(kmeans))]
        line_p1, line_p2 = (0, scores[0]), (max_clusters, scores[-1])

        if self.verbose:
            print('Calculating Elbow point...')

        distances = []
        for n_score, score in enumerate(scores):
            point = (n_score, score)
            distances.append(self.distance_point_line(line_p1, line_p2, point))

        return distances.index(max(distances)), scores

    @staticmethod
    def distance_point_line(line_p1, line_p2, point) -> float:
        """
        Calculates the distance between the line going from p1 to p2 and the given point.

        Args:
            line_p1 (tuple): line point 1 (x, y)
            line_p2 (tuple): line point 2 (x, y)
            point (tuple): point (x, y)

        Returns:
            float: distance between the line and the point.
        """
        p1 = np.asarray(line_p1)
        p2 = np.asarray(line_p2)
        p = np.asarray(point)
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p)) / \
            np.linalg.norm(p2 - p1)

        return d

    @staticmethod
    def get_distance_by_point(data, clusters_centers, labels) -> List[float]:
        """
        Calculates the distance between a data point and its assigned cluster centroid.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data points.
            clusters_centers (numpy.ndarray): cluster centroids.
            labels (numpy.ndarray): assigned cluster to every data point.

        Returns:
            list of float: distances.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        distances = []
        for i in range(0, len(data)):
            xa = np.array(data[i])
            xb = clusters_centers[labels[i] - 1]
            distances.append(np.linalg.norm(xa - xb))

        return distances


class AnomalyOneClassSVM(AnomalyModel):
    """
    Unsupervised anomaly detection model based on One Class Support Vector Machine. The model is trained with data that
    doesn't contains anomalies. The idea of this model is to find a function that is positive for regions with high
    density of points (not anomaly), and negative for small densities (anomaly).

    Args:
        outliers_fraction (float): outliers fraction. Defaults to 0.01 (3 standard deviations).
        gamma (float): kernel coefficient. Defaults to 0.01.
        verbose (bool): verbose mode. Defaults to False.
    """

    def __init__(self, outliers_fraction=0.01, gamma=0.01, verbose=False):
        super().__init__()
        self._outliers_fraction = outliers_fraction
        self._gamma = gamma
        self._svm = OneClassSVM(nu=self._outliers_fraction, kernel='rbf', gamma=self._gamma)
        self.verbose = verbose

    def train(self, data) -> None:
        """
        Trains the One Class Support Vector Machine Model.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data
        """
        if self.verbose:
            print('Training the OneClassSVM model...')

        self._svm.fit(data)

    def predict(self, data) -> np.ndarray:
        """
        Predicts if the data point belongs to the positive region or the negative region.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data

        Returns:
            numpy.ndarray: scores
        """
        return self._svm.predict(data)

    def flag_anomaly(self, data) -> List[bool]:
        """
        Flags as anomaly or not the data points.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data

        Returns:
            list of bool: list of bool telling if a data point is an anomaly or not.
        """
        predictions = self.predict(data)
        return predictions == -1


class AnomalyGaussianDistribution(AnomalyModel):
    """
    Unsupervised anomaly detection model based on Gaussian Distribution. The model is trained with data that doesn't
    contains anomalies. This model computes the mean and the variance for every feature in the dataset and then with
    these values calculates the probability of a data point to belong to the distribution.

    Args:
        verbose (bool): verbose mode. Defaults to False.
    """

    def __init__(self, verbose=False):
        super().__init__()
        self._mean = None
        self._variance = None
        self._epsilon = None
        self._probabilities = None
        self.verbose = verbose

    def train(self, data, labels=None) -> None:
        """
        Trains the model with the data passed. For that, the mean and the variance are calculated for every feature in
        the data.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data.
            labels (numpy.ndarray or pandas.DataFrame): labels of training data. Defaults to None.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if isinstance(labels, pd.DataFrame):
            labels = labels.values

        # all data is healthy, so ones
        if labels is None:
            labels = np.ones((data.shape[0],))

        (self._mean, self._variance) = self.estimate_parameters(data)
        self._probabilities = self.calculate_probability(data)
        (self._epsilon, _) = self.establish_threshold(labels, self._probabilities)

    def predict(self, data) -> np.ndarray:
        """
        Calculates the probability of a data point to belong to the Gaussian Distribution.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data

        Returns:
            numpy.ndarray: probabilities.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        return self.calculate_probability(data)

    def flag_anomaly(self, data) -> np.ndarray:
        """
        Flag the data point as an anomaly if the probability surpass epsilon (threshold).

        Args:
            data (numpy.ndarray or pandas.DataFrame): data to flag as an anomaly or not.

        Returns:
            numpy.ndarray: list of bool telling if a data point is an anomaly or not.
        """
        probabilities = self.predict(data)
        return probabilities < self._epsilon

    def calculate_probability(self, data) -> np.ndarray:
        """
        Calculates the probability of a list of data points to belong to the Gaussian Distribution.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        num_samples = data.shape[0]
        num_features = data.shape[1]

        probabilities = np.ones((num_samples, 1))

        # calculate p(x)
        for sample_index in range(num_samples):
            for feature_index in range(num_features):
                # power of e
                power_dividend = np.power(data[sample_index, feature_index] - self._mean[feature_index], 2)
                power_divider = 2 * self._variance[feature_index]
                e_power = -1 * power_dividend / power_divider

                # prefix multiplier
                prefix_multiplier = 1 / np.sqrt(2 * np.pi * self._variance[feature_index])

                # probability
                probability = prefix_multiplier * np.exp(e_power)
                probabilities[sample_index] *= probability

        return probabilities.reshape(-1, )

    @staticmethod
    def estimate_parameters(data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes mu (mean) and sigma^2 (variance) values for a data distribution.

        Args:
            data (numpy.ndarray): data

        Returns:
            tuple:
                mu (numpy.ndarray): mean,
                sigma^2 (numpy.ndarray): variance
        """
        num_samples = data.shape[0]
        mu = np.mean(data, axis=0)
        sigma_squared = np.power(np.std(data, axis=0), 2)

        return mu, sigma_squared

    @staticmethod
    def establish_threshold(labels, probabilities) -> Tuple[float, float]:
        """
        Calculates the threshold for defining an anomaly.

        Args:
            labels (numpy.ndarray): label for each probability.
            probabilities (numpy.ndarray): probabilities of belonging to the Gaussian Distribution.

        Returns:
            tuple:
                best_epsilon (float): threshold,
                best_f1 (float): best f1 score
        """
        best_epsilon = 0
        best_f1 = 0

        min_prob = min(probabilities)
        max_prob = max(probabilities)
        step_size = (max_prob - min_prob) / 1000

        for epsilon in np.arange(min_prob, max_prob, step_size):
            predictions = probabilities < epsilon

            false_positives = np.sum((predictions == 1) & (labels == 0))
            false_negatives = np.sum((predictions == 0) & (labels == 1))
            true_positives = np.sum((predictions == 1) & (labels == 1))

            # prevent division by 0
            if (true_positives + false_positives) == 0 or (true_positives + false_negatives) == 0:
                continue

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1 = 2 * precision * recall / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_epsilon = epsilon

            return best_epsilon, best_f1


class AnomalyIsolationForest(AnomalyModel):
    """
    Unsupervised anomaly detection model based on One Class Support Vector Machine. The model is trained with data that
    doesn't contains anomalies. The Isolation Forest algorithm isolates observations by randomly selecting a feature and
    then randomly selecting a split value between the maximum and the minimum values of the selected feature. Isolating
    anomalies is easier because only a few conditions are needed to separate them from normal values.

    Args:
        outliers_fraction (float): outliers fraction. Defaults to 0.01 (3 standard deviations).
        verbose (bool): verbose mode. Defaults to False.
    """

    def __init__(self, outliers_fraction=0.01, verbose=False):
        super().__init__()
        self._forest = IsolationForest(contamination=outliers_fraction, behaviour='new')
        self._outliers_fraction = outliers_fraction
        self.verbose = verbose

    def train(self, data) -> None:
        """
        Trains the Isolation Forest Model.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data
        """
        if self.verbose:
            print('Training the Isolation Forest model...')

        self._forest.fit(data)

    def predict(self, data) -> np.ndarray:
        """
        Predicts if the data points are anomalies or inliers.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data to predict.

        Returns:
            numpy.ndarray: scores.
        """
        return self._forest.predict(data)

    def flag_anomaly(self, data) -> np.ndarray:
        """
        Flag a data point as an anomaly or as an inlier. If the score from the predict method is negative, then it's an
        anomaly, if it's positive then it's an inlier.

        Returns:
            numpy.ndarray: list containing bool values telling if data point is an anomaly or not.
        """
        scores = self.predict(data)
        return scores < 0
