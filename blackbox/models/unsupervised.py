from typing import List, Tuple
import numpy as np
import pandas as pd
from blackbox.models.base_model import AnomalyModel


class AnomalyPCAMahalanobis(AnomalyModel):
    """
    Unsupervised anomaly detection model based on Primary Component Analysis and
    Mahalanobis Distance. The model is trained with data that is considered to not have
    anomalies (normal operation) or with data in which the contamination proportion is
    known, and reduces the dimensionality of the data with PCA, which is a technique
    very sensible to outliers. When the model receives a new data point, it will reduce
    the dimensionality of the point with PCA and then will calculate the distance of
    this point to the train distribution. If the distance surpass a threshold previously
    established, then the data point is flagged as an anomaly.

    Args:
        n_components (int or float): number of components to which the data have to be
            reduced. Defaults to 2.
        contamination (float): contamination fraction of training dataset. Defaults to
            0.1.

        verbose (bool): verbose mode. Defaults to False.

    Todo:
        * Add n_jobs argument
    """

    from sklearn.decomposition import PCA

    def __init__(self, n_components=2, contamination=0.1, verbose=False) -> None:
        super().__init__()
        self._pca = self.PCA(n_components=n_components, svd_solver="full")
        self._data = None
        self._distances = None
        self._cov = None
        self._inv_cov = None
        self._mean_data = None
        self._threshold = None
        self._contamination = contamination
        self._verbose = verbose

    def train(self, data) -> None:
        """
        Trains the model with the train data given.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        self._data = self._pca.fit_transform(data)
        self._distances = self.mahalanobis_distance(self._data)
        self._threshold = self.calculate_threshold()

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
        Flag the data points as anomaly if the calculated Mahalanobis distance surpass a
        established threshold.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data to flag as anomalous or not
                anomalous.

        Returns:
            numpy.ndarray: list of booleans telling if point is anomalous or not.
        """
        distances = self.predict(data)
        return distances > self._threshold

    def calculate_threshold(self) -> float:
        """
        Computes the threshold that has to surpass a distance of a point to be flagged
        as an anomaly.

        Returns:
            float: threshold.
        """
        threshold = np.percentile(self._distances, 100 * (1 - self._contamination))
        return threshold

    def mahalanobis_distance(self, x) -> np.ndarray:
        """
        Computes the Mahalanobis distance between each row of x and the data
        distribution.

        Args:
            x (numpy.ndarray or pandas.DataFrame): vector or matrix of data with p
                columns.

        Returns:
            numpy.ndarray: distance between each row of x and the data distribution.
        """
        if isinstance(x, pd.DataFrame):
            x = x.values

        if self._cov is None:
            self._cov = np.cov(self._data, rowvar=False)
            self._inv_cov = np.linalg.inv(self._cov)

        if self._mean_data is None:
            self._mean_data = np.mean(self._data)

        distances = []
        for point in x:
            diff = point - self._mean_data
            distances.append(np.sqrt(np.dot(np.dot(diff, self._inv_cov), diff.T)))

        return np.array(distances)


class AnomalyAutoencoder(AnomalyModel):
    """
    Unsupervised anomaly detection model based on a Deep Neural Network of Autoencoder
    type. The model is trained with data that doesn't contains anomalies or with data in
    which the contamination proportion is known. This kind of DNN compress or reduce the
    data from the input and then amplify or reconstruct the data to the original size at
    the output. This way, the DNN will be generating new data very similar to the
    original data. This characteristic is used to train the Autoencoder with data
    without anomalies. The Autoencoder will be able to reconstruct data similar to the
    training data (value of loss function will be low) and won't be able to reconstruct
    data with anomalies (value of loss function high).

    Args:
        hidden_neurons (list): hidden layers and the number of neurons for each hidden
            layer. Defaults to [32, 16, 16, 32].
        dropout_rate (float): dropout rate across all layers. Float between 0 and 1.
            Defaults to 0.2.
        activation (str): activation function that layers will have. Defaults to 'elu'.
        kernel_initializer (str): kernel initializer that layers will have. Defaults to
            'glorot_uniform'.
        kernel_regularizer (keras.regularizers): kernel regularizer that layers will
            have. Defaults to None.
        loss_function (str): loss function that the Autoencoder will have. Defaults to
            'mse'.
        optimizer (str): optimizer that the Autoencoder will have. Defaults to 'adam'.
        epochs (int): number of times that all the batches will be processed during the
            Autoencoder training. Defaults to 100.
        batch_size (int): batch size. Defaults to 10.
        validation_split (float): percentage of the training data that will be used for
            model validation. Defaults to 0.05.
        contamination (float): contamination fraction of the training dataset. Defaults
            to 0.1.
        early_stopping (boolean or int): indicates if early stopping is going to be used
            in training process. This can speed up the training process stopping this
            process when the validation_loss is not improving. If the value of this
            argument is an integer, it will define the number of epochs with no improve
            in validation_loss before stopping the training process. Defaults to False.
        verbose (bool): verbose mode. Defaults to False.

    References:
        * An, J., & Cho, S. (2015). Variational autoencoder based anomaly detection
            using reconstruction probability. Special Lecture on IE, 2(1).
    """

    from keras import models
    from keras.layers import Dense, Dropout
    from keras import regularizers
    from keras.callbacks.callbacks import EarlyStopping

    def __init__(
        self,
        hidden_neurons=None,
        dropout_rate=0.2,
        activation="elu",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        loss_function="mse",
        optimizer="adam",
        epochs=100,
        batch_size=10,
        validation_split=0.05,
        contamination=0.1,
        early_stopping=False,
        verbose=False,
    ):
        super().__init__()

        # model parameters
        self._hidden_neurons = hidden_neurons
        self._dropout_rate = dropout_rate
        self._activation = activation
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._loss_function = loss_function
        self._optimizer = optimizer

        # training parameters
        self._epochs = epochs
        self._batch_size = batch_size
        self._validation_split = validation_split
        self._contamination = contamination
        self._early_stopping = early_stopping

        # default values
        if self._kernel_regularizer is None:
            self._kernel_regularizer = self.regularizers.l2(0.0)

        if self._hidden_neurons is None:
            self._hidden_neurons = [32, 16, 16, 32]

        # Verify that the network design is symmetric
        if not self._hidden_neurons == self._hidden_neurons[::-1]:
            raise ValueError(
                "Hidden neurons should be symmetric: {}".format(self._hidden_neurons)
            )

        self._autoencoder = None
        self._history = None
        self._loss = None
        self._threshold = None
        self._verbose = verbose

    def train(self, data) -> None:
        """
        Trains the Autoencoder.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        # Verify that number of neurons doesn't exceeds the number of features
        self._n_features = data.shape[1]
        if self._n_features < min(self._hidden_neurons):
            raise ValueError(
                "Number of neurons should not exceed the number of features."
            )

        if self._verbose:
            verbosity_level = 1
        else:
            verbosity_level = 0

        cb_list = []
        if not isinstance(self._early_stopping, bool):
            if self._verbose:
                print(
                    "Using EarlyStopping in trainning process. Number of epochs without"
                    " improvement before stopping training process: {}".format(
                        self._early_stopping
                    )
                )
            es = self.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=self._early_stopping,
                verbose=verbosity_level,
            )
            cb_list.append(es)

        self._autoencoder = self.build_autoencoder()
        self.history = self._autoencoder.fit(
            x=data,
            y=data,
            batch_size=self._batch_size,
            epochs=self._epochs,
            validation_split=self._validation_split,
            shuffle=True,
            verbose=verbosity_level,
            callbacks=cb_list,
        )
        predict = self._autoencoder.predict(data)
        self._loss = self.mean_absolute_error(data, predict)
        self._threshold = self.establish_threshold()

    def predict(self, data) -> np.ndarray:
        """
        Tries to reconstruct the input.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data

        Returns:
            numpy.ndarray: reconstructed inputs.
        """
        reconstructed_data = self._autoencoder.predict(data)
        return reconstructed_data

    def flag_anomaly(self, data) -> np.ndarray:
        """
        Flag a data point as an anomaly if the MAE (Mean Absolute Error) is higher than
        the established threshold.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data

        Returns:
            numpy.ndarray: list containing bool values telling if data point is an
                anomaly or not.
        """
        predict = self._autoencoder.predict(data)
        loss = self.mean_absolute_error(data, predict)
        return loss > self._threshold

    def build_autoencoder(self):
        """
        Builds the model the Autoencoder model.

        Returns:
            keras.engine.sequential.Sequential: Autoencoder model.
        """
        model = self.models.Sequential()

        # input layer
        model.add(
            self.Dense(
                units=self._hidden_neurons[0],
                activation=self._activation,
                kernel_initializer=self._kernel_initializer,
                activity_regularizer=self._kernel_regularizer,
                input_shape=(self._n_features,),
            )
        )
        model.add(self.Dropout(self._dropout_rate))

        # hidden layers
        for _, hidden_neurons in enumerate(self._hidden_neurons, 2):
            model.add(
                self.Dense(
                    units=hidden_neurons,
                    activation=self._activation,
                    kernel_initializer=self._kernel_initializer,
                    activity_regularizer=self._kernel_regularizer,
                )
            )
            model.add(self.Dropout(self._dropout_rate))

        # output layer
        model.add(
            self.Dense(
                units=self._n_features, kernel_initializer=self._kernel_initializer
            )
        )

        # compile
        model.compile(optimizer=self._optimizer, loss=self._loss_function)

        if self._verbose:
            print(model.summary())

        return model

    def establish_threshold(self) -> float:
        """
        Computes the threshold that has to surpass the calculated loss (MAE) of a point
        to be flagged as an anomaly.

        Returns:
            float: threshold.
        """
        threshold = np.percentile(self._loss, 100 * (1 - self._contamination))
        return threshold

    @staticmethod
    def mean_absolute_error(data, predicted_data) -> np.ndarray:
        """
        Calculates the MAE (Mean Absolute Error).

        Args:
            data (numpy.ndarray or pandas.DataFrame): real data.
            predicted_data (numpy.ndarray or pandas.DataFrame): predicted data.

        Returns:
            numpy.ndarray: mean absolute error.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if isinstance(predicted_data, pd.DataFrame):
            predicted_data = predicted_data.values

        mae = np.mean(np.abs(predicted_data - data), axis=1)
        return mae


class AnomalyKMeans(AnomalyModel):
    """
    Unsupervised anomaly detection model based on K-Means Clustering. The model is
    trained with data that doesn't contain anomalies or data with anomalies in which the
    contamination proportion is known. This will create 'k' similar clusters of data
    points. When new data points are received, we will predict the nearest cluster in
    which the data belongs, and then calculate the distance from the new data points to
    the cluster centroid. If this distance surpass a established threshold, then the
    data point will be flagged as an anomaly.

    Args:
        n_clusters (Int): indicates the number of clusters. Defaults to None.
        contamination (float): contamination (float): contamination fraction of the
            training dataset. Defaults to 0.1.
        max_cluster_elbow (int): maximum number of cluster to test in the Elbow Method.
            Defaults to 100.
        n_jobs (int): number of cores to use in training and predicting process.
            Defaults to 1.
        verbose (bool): verbose mode. Defaults to False.

    Todo:
        * Add n_jobs argument
    """

    from sklearn.cluster import KMeans

    def __init__(
        self,
        n_clusters=None,
        contamination=0.1,
        max_cluster_elbow=100,
        n_jobs=1,
        verbose=False,
    ):
        super().__init__()
        self._verbose = verbose
        self._kmeans = None
        self._n_clusters = n_clusters
        self._contamination = contamination
        self._max_cluster_elbow = max_cluster_elbow
        self._n_jobs = n_jobs
        self._threshold = None
        self._distances = None

    def train(self, data) -> None:
        """
        Trains the model with the train data given. First, the optimal number of
        clusters is calculated with the Elbow Method. Then, a K-Means Cluster model is
        fitted with the optimal number of clusters.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if self._n_clusters is None:
            if self._verbose:
                print("Calculating optimal number of clusters with Elbow Method...")

            (self._n_clusters, _) = self.elbow(data, self._max_cluster_elbow)

            if self._verbose:
                print(
                    "Optimal number of n_clusters: {}. Fitting model...".format(
                        self._n_clusters
                    )
                )

        self._kmeans = self.KMeans(n_clusters=self._n_clusters, n_jobs=self._n_jobs)
        self._kmeans.fit(data)
        self._distances = self.get_distance_by_point(
            data, self._kmeans.cluster_centers_, self._kmeans.labels_
        )

        if self._verbose:
            print("Calculating threshold value to flag an anomaly...")

        self._threshold = self.calculate_threshold()

    def predict(self, data) -> List[float]:
        """
        Calculates the cluster of each data point and then calculates the distance
        between the data point and the cluster centroid.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data

        Returns:
            list of float: distances between the data given and its assigned cluster
                centroid.
        """
        data_labels = self._kmeans.predict(data)
        distances = self.get_distance_by_point(
            data, self._kmeans.cluster_centers_, data_labels
        )
        return distances

    def flag_anomaly(self, data) -> np.ndarray:
        """
        Flag the data points as anomaly if the calculated distance between the point and
        its assigned cluster centroid surpass a established threshold.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data to flag as anomalous or not
                anomalous.

        Returns:
            numpy.ndarray: list of booleans telling if point is anomalous or not.
        """
        distances = self.predict(data)
        return distances > self._threshold

    def calculate_threshold(self) -> float:
        """
        Calculates the threshold that has to surpass a distance between a data point and
        its cluster to be flagged as an anomaly.

        Returns:
            float: threshold value.
        """
        threshold = np.percentile(self._distances, 100 * (1 - self._contamination))
        return threshold

    def elbow(self, data, max_clusters=100) -> tuple:
        """
        Takes training data and computes K-Means model for each number of n_clusters
        given and get the score of each K-Means model. Then, calculate the distance
        between the line that goes from the point of the first score to the point of the
        last score with each score of the K-Means models.

        Args:
            data (pandas.DataFrame): training data.
            max_clusters (int): number of maximum clusters.

        Returns:
            tuple: tuple containing:
                n_cluster (float): optimal number of clusters (Elbow point)
                scores (list): list of float with scores for every K-Means model.
        """
        if self._verbose:
            print("Computing K-Means models...")

        n_clusters = range(1, max_clusters)
        kmeans = [
            self.KMeans(n_clusters=i, n_jobs=self._n_jobs).fit(data) for i in n_clusters
        ]
        scores = [kmeans[i].score(data) for i in range(len(kmeans))]
        line_p1, line_p2 = (0, scores[0]), (max_clusters, scores[-1])

        if self._verbose:
            print("Calculating Elbow point...")

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
        d = np.linalg.norm(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)

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
    Unsupervised anomaly detection model based on One Class Support Vector Machine. The
    model is trained with data that doesn't contain anomalies or data with anomalies in
    which the contamination proportion is known. The idea of this model is to find a
    function that is positive for regions with high density of points (not anomaly), and
    negative for small densities (anomaly).

    Args:
        contamination (float): contamination fraction of training dataset. Defaults to
            0.1.
        kernel (str): kernel type. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
            ‘precomputed’ or a callable. Defaults to 'rbf'.
        degree (int): degree of the polynomial kernel function ('poly). Ignored by all
            other kernels. Defaults to 3.
        gamma (str or float): kernel coefficient for 'rbf', 'poly' and 'sigmod'.
            Available values are 'scale', 'auto' or float. Defaults to 'scale'.
        coef0 (float): independent term in kernel function. Only significant in 'poly'
            and 'sigmoid'. Defaults to 0.0.
        tol (float): tolerance for stopping criterion. Defaults to 0.001.
        shrinking (bool): wheter to use the shrinking heuristic. Defaults to True.
        cache_size (float): specify the size of the kernel cache in MB. Defaults to 200.
        n_jobs (int): number of cores to use in training and predicting process.
            Defaults to 1.
        verbose (bool): verbose mode. Defaults to False.

    Todo:
        * Add n_jobs argument
    """

    from sklearn.svm import OneClassSVM

    def __init__(
        self,
        contamination=0.1,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        tol=0.001,
        shrinking=True,
        cache_size=200,
        n_jobs=1,
        verbose=False,
    ):
        super().__init__()
        self._contamination = contamination
        self._svm = self.OneClassSVM(
            nu=self._contamination,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            cache_size=cache_size,
            tol=tol,
            n_jobs=n_jobs,
        )
        self._verbose = verbose

    def train(self, data) -> None:
        """
        Trains the One Class Support Vector Machine Model.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if self._verbose:
            print("Training the OneClassSVM model...")

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
    Unsupervised anomaly detection model based on Gaussian Distribution. The model is
    trained with data that doesn't contains anomalies. This model computes the mean and
    the variance for every feature in the dataset and then with these values calculates
    the probability of a data point to belong to the distribution.

    Args:
        verbose (bool): verbose mode. Defaults to False.
    """

    def __init__(self, verbose=False):
        super().__init__()
        self._mean = None
        self._variance = None
        self._epsilon = None
        self._probabilities = None
        self._verbose = verbose

    def train(self, data, labels=None) -> None:
        """
        Trains the model with the data passed. For that, the mean and the variance are
        calculated for every feature in the data.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data.
            labels (numpy.ndarray or pandas.DataFrame): labels of training data.
                Defaults to None.
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
        Calculates the probability of a data point to belong to the Gaussian
        Distribution.

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
        Calculates the probability of a list of data points to belong to the Gaussian
        Distribution.

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
                power_dividend = np.power(
                    data[sample_index, feature_index] - self._mean[feature_index], 2
                )
                power_divider = 2 * self._variance[feature_index]
                e_power = -1 * power_dividend / power_divider

                # prefix multiplier
                prefix_multiplier = 1 / np.sqrt(
                    2 * np.pi * self._variance[feature_index]
                )

                # probability
                probability = prefix_multiplier * np.exp(e_power)
                probabilities[sample_index] *= probability

        return probabilities.reshape(-1,)

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
        mu = np.mean(data, axis=0)
        sigma_squared = np.power(np.std(data, axis=0), 2)

        return mu, sigma_squared

    @staticmethod
    def establish_threshold(labels, probabilities) -> Tuple[float, float]:
        """
        Calculates the threshold for defining an anomaly.

        Args:
            labels (numpy.ndarray): label for each probability.
            probabilities (numpy.ndarray): probabilities of belonging to the Gaussian
                Distribution.

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
            if (true_positives + false_positives) == 0 or (
                true_positives + false_negatives
            ) == 0:
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
    Unsupervised anomaly detection model based on Isolation Forest. The model is trained
    with data that doesn't contains anomalies. The Isolation Forest algorithm isolates
    observations by randomly selecting a feature and then randomly selecting a split
    value between the maximum and the minimum values of the selected feature. Isolating
    anomalies is easier because only a few conditions are needed to separate them from
    normal values.

    Args:
        contamination (float): contamination fraction in dataset. Defaults to 0.1.
        n_estimators (int): the number of base estimators in the ensemble. Defaults to
            100.
        max_features (int or float): number of features to draw from X to train each
            base estimator. Defaults to 1.0.
        bootstrap (bool): if True, individual trees are fit on random subsets of the
            training data sampled with replacement. If False, sampling without
            replacement is permformed. Defaults to False.
        n_jobs (int): number of cores to use in training and predicting process.
            Defaults to 1.
        verbose (bool): verbose mode. Defaults to False.

    Todo:
        * Add n_jobs argument
    """

    from sklearn.ensemble import IsolationForest

    def __init__(
        self,
        contamination=0.1,
        n_estimators=100,
        max_features=1.0,
        bootstrap=False,
        n_jobs=1,
        verbose=False,
    ):
        super().__init__()
        self._forest = self.IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=bootstrap,
            behaviour="new",
            n_jobs=n_jobs,
        )
        self._contamination = contamination
        self._verbose = verbose
        self._training_outliers = None

    def train(self, data) -> None:
        """
        Trains the Isolation Forest Model.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if self._verbose:
            print("Training the Isolation Forest model...")

        self._training_outliers = self._forest.fit_predict(data) < 0

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
        Flag a data point as an anomaly or as an inlier. If the score from the predict
        method is negative, then it's an anomaly, if it's positive then it's an inlier.

        Returns:
            numpy.ndarray: list containing bool values telling if data point is an
                anomaly or not.
        """
        scores = self.predict(data)
        return scores < 0


class AnomalyLOF(AnomalyModel):
    """
    Unsupervised anomaly detection model based using the Local Outlier Factor model.

    Args:
        contamination (float): contamination fraction in dataset. Defaults to 0.1.
        n_neighbors (int): number of neighbors to use by default for kneighbors queries.
            Defaults to 20.
        algorithm (string): algorithm used to compute the nearest neighbors. Available
            algorithms are 'ball_tree', 'kd_tree', 'brute' or 'auto'. Defaults to 'auto'.
        leaf_size (int): leaf size passed to BallTree or KDTree. Defaults to 30.
        metric (string): the distance metric to use for the tree. Defaults to
            'minkowski'.
        p (int): parameter for the Minkowski metric. If p = 1, then it's equivalent to
            using Manhattan distance, and if p = 2 then it's equivalent to use Euclidean
            distance. Defaults to 2.
        n_jobs (int): number of cores to use in training and predicting process.
            Defaults to 1.
        verbose (bool): verbose mode. Defaults to False.
    """

    from sklearn.neighbors import LocalOutlierFactor

    def __init__(
        self,
        contamination=0.1,
        n_neighbors=20,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        n_jobs=1,
        verbose=False,
    ):
        self._contamination = contamination
        self._lof = self.LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            novelty=True,
            n_jobs=n_jobs,
        )
        self._verbose = verbose

    def train(self, data):
        """
        Trains the Local Outlier Factor model.

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if self._verbose:
            print("Training the Local Outlier Factor model...")

        self._lof.fit(data)

    def predict(self, data):
        """
        Predicts if the data points are anomalies or inliers.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data to predict.

        Returns:
            numpy.ndarray: scores.
        """
        return self._lof.predict(data)

    def flag_anomaly(self, data):
        """
        Flag a data point as an anomaly or as an inlier. If the score from the predict
        method is negative, then it's an anomaly, if it's positive then it's an inlier.

        Returns:
            numpy.ndarray: list containing bool values telling if data point is an
                anomaly or not.
        """
        scores = self.predict(data)
        return scores < 0


class AnomalyKNN(AnomalyModel):
    """
    Unsupervised anomaly detection model based on k-Nearest Neighbor. The model is
    trained with data that doesn't contain anomalies or data with anomalies in which the
    contamination proportion is known. The outlier score can be calculated using the
    distance to kth neighbor, the average of k neighbors or the median of the distance
    of k neighbors.

    Args:
        n_neighbors (int): number of neighbors to use. Defaults to 5.
        radius (float): range of parameter space to use by default for radius_neighbors
            queries. Defaults to 1.0.
        algorithm (string): algorithm used to compute the nearest neighbors. Available
            algorithms are 'ball_tree', 'kd_tree', 'brute' or 'auto'. Defaults to 'auto'.
        leaf_size (int): leaf size passed to BallTree or KDTree. Defaults to 30.
        metric (string): the distance metric to use for the tree. Defaults to
            'minkowski'.
        p (int): parameter for the Minkowski metric. If p = 1, then it's equivalent to
            using Manhattan distance, and if p = 2 then it's equivalent to use Euclidean
            distance. Defaults to 2.
        contamination (float): contamination fraction in dataset. Defaults to 0.1.
        score_func (string): the function used to score anomalies. Available scores
            are 'max_distance', 'average' or 'median'. Defaults to 'distance'.
        n_jobs (int): number of cores to use in training and predicting process.
            Defaults to 1.
        verbose (bool): verbose mode. Defaults to False.

    References:
        * Ramaswamy, S., Rastogi, R., & Shim, K. (2000, May). Efficient algorithms for
            mining outliers from large data sets. In ACM Sigmod Record (Vol. 29, No. 2,
            pp. 427-438). ACM.
        * Angiulli, F., & Pizzuti, C. (2002, August). Fast outlier detection in high
            dimensional spaces. In European Conference on Principles of Data Mining and
            Knowledge Discovery (pp. 15-27). Springer, Berlin, Heidelberg.
    """

    from sklearn.neighbors import NearestNeighbors

    def __init__(
        self,
        n_neighbors=5,
        radius=1.0,
        leaf_size=30,
        metric="minkowski",
        p=2,
        algorithm="auto",
        score_func="max_distance",
        contamination=0.1,
        n_jobs=1,
        verbose=False,
    ):
        self._n_neighbors = n_neighbors
        self._score_func = score_func
        self._contamination = contamination
        self._distances = None
        self._knn = self.NearestNeighbors(
            n_neighbors=n_neighbors,
            radius=radius,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            algorithm=algorithm,
            n_jobs=n_jobs,
        )
        self._distances = None
        self._threshold = None
        self._verbose = verbose

    def train(self, data):
        """
        Trains the Nearest Neighbors

        Args:
            data (numpy.ndarray or pandas.DataFrame): training data.
        """
        if isinstance(data, pd.DataFrame):
            data = data.values

        if self._verbose:
            print("Training the k-Nearest Neighbors model...")

        self._knn.fit(data)
        distances, _ = self._knn.kneighbors(
            n_neighbors=self._n_neighbors, return_distance=True
        )
        self._distances = self.get_dist_by_score_func(distances)
        self._threshold = self.calculate_threshold()

    def predict(self, data):
        """
        Calculates the distance of data points to its k-neighbors and then the outlier
        score is calculated.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data to calculate its outlier
                score.

        Returns:
            np.ndarray: array with the outlier score of the data points.
        """
        n_samples = data.shape[0]
        scores = np.zeros((n_samples, 1))

        for i in range(n_samples):
            sample_features = np.array([data[i]])
            distances, _ = self._knn._tree.query(sample_features, k=self._n_neighbors)
            score = self.get_dist_by_score_func(distances)
            scores[i] = score

        return scores.ravel()

    def flag_anomaly(self, data) -> np.ndarray:
        """
        Flags as anomaly or not the data points.

        Args:
            data (numpy.ndarray or pandas.DataFrame): data.

        Returns:
            np.ndarray: list of bool telling if a data point is an anomaly or not.
        """
        scores = self.predict(data)
        return scores > self._threshold

    def get_dist_by_score_func(self, distances) -> np.ndarray:
        """
        Calculates the outlier score of the distances.

        Args:
            distances (np.ndarray): array with the distances of data point to its
                k-neighbors.

        Returns:
            np.ndarray: outlier scores.
        """
        scores = None
        if self._score_func == "max_distance":
            scores = distances[:, -1]
        elif self._score_func == "average":
            scores = np.mean(distances, axis=1)
        elif self._score_func == "median":
            scores = np.median(distances, axis=1)

        return scores

    def calculate_threshold(self):
        """
        Calculates the threshold that has to surpass a distance between a data point and
        its cluster to be flagged as an anomaly.

        Returns:
            float: threshold value.
        """
        threshold = np.percentile(self._distances, 100 * (1 - self._contamination))
        return threshold
