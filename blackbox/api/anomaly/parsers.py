from flask_restplus.reqparse import RequestParser
from werkzeug.datastructures import FileStorage

# Train parser for '/api/v1/train/{entity}' endpoint
train_parser = RequestParser()

# Model name
train_parser.add_argument(
    "name",
    type=str,
    help="Optional name to identify the Blackbox model that will be trained.",
)

# List of models
train_parser.add_argument(
    "models",
    type=str,
    choices=(
        "PCAMahalanobis",
        "Autoencoder",
        "KMeans",
        "OneClassSVM",
        "GaussianDistribution",
        "IsolationForest",
    ),
    help="List of the models that are going to be inside the Blackbox separated by a"
    " comma. If not specified, then every anomaly detection model available will be used.",
)

# Train file
train_parser.add_argument(
    "file", type=FileStorage, required=True, location="files", help="CSV training file"
)
train_parser.add_argument(
    "input_arguments",
    type=str,
    required=True,
    help="List of input arguments for Anomaly Detection models separated by a comma.",
)

# Train file features
train_parser.add_argument(
    "contamination",
    type=float,
    default=0.1,
    help="Contamination fraction in training dataset.",
)

# PCA + Mahalanobis params
train_parser.add_argument(
    "n_components",
    type=int,
    default=2,
    help="Numbers of components for the PCA technique.",
)

# Autoencoder params
train_parser.add_argument(
    "hidden_neurons",
    type=int,
    default=[32, 16, 16, 32],
    action="append",
    help="Hidden layers and the number of neurons in each layer for the Autoencoder"
    " model. Example: 32,16,16,32.",
)
train_parser.add_argument(
    "dropout_rate",
    type=float,
    default=0.2,
    help="Dropout rate across all the layers of the Autoencoder.",
)
train_parser.add_argument(
    "activation",
    type=str,
    default="elu",
    choices=(
        "elu",
        "softmax",
        "selu",
        "softplus",
        "softsign",
        "relu",
        "tanh",
        "sigmoid",
        "hard_sigmoid",
        "exponential",
    ),
    help="Layers activation function of Autoencoder.",
)
train_parser.add_argument(
    "kernel_initializer",
    type=str,
    default="glorot_uniform",
    choices=(
        "Zeros",
        "Ones",
        "Constant",
        "RandomNormal",
        "RandomUniform",
        "TruncatedNormal",
        "VarianceScaling",
        "Orthogonal",
        "Identity",
        "lecun_uniform",
        "glorot_normal",
        "glorot_uniform",
        "he_normal",
        "lecun_normal",
        "he_uniform",
    ),
    help="Layers kernel initializer of Autoencoder.",
)
train_parser.add_argument(
    "loss_function", type=str, default="mse", help="Loss funcion of Autoencoder."
)
train_parser.add_argument(
    "optimizer",
    type=str,
    default="adam",
    choices=("sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"),
    help="Autoencoder optimizer.",
)
train_parser.add_argument(
    "epochs",
    type=int,
    default=100,
    help="Number of times that all the batches will be processed in the Autoencoder",
)
train_parser.add_argument("batch_size", type=int, default=32, help="Batch size")
train_parser.add_argument(
    "validation_split",
    type=float,
    default=0.05,
    help="Percentage of the training data that will be used for validation in the"
    " Autoencoder",
)
train_parser.add_argument(
    "early_stopping",
    type=bool,
    default=False,
    choices=(True, False),
    help="Stops the training process in Autoencoder when it's not getting any"
    "improvement which saves time.",
)

# K-Means params
train_parser.add_argument(
    "n_clusters", type=int, default=None, help="Number of cluster for the K-Means."
)

# OCSVM params
train_parser.add_argument(
    "kernel",
    type=str,
    default="rbf",
    choices=("linear", "poly", "rbf", "sigmoid", "precomputed"),
    help="Kernel type for One Class SVM.",
)
train_parser.add_argument(
    "degree",
    type=int,
    default=3,
    help="Degree of the polynomial kernel function in One Class SVM.",
)
train_parser.add_argument(
    "gamma",
    type=str,
    default="scale",
    help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid' in One Class SVM. It can"
    "'scale', 'auto' or float.",
)
train_parser.add_argument(
    "coef0",
    type=float,
    default=0.0,
    help="Independent term in kernel function of One Class SVM. Only significant in"
    " 'poly'",
)
train_parser.add_argument(
    "tol",
    type=float,
    default=0.001,
    help="Tolerance for stopping criterion in One Class SVM.",
)
train_parser.add_argument(
    "shrinking",
    type=bool,
    default=True,
    choices=(True, False),
    help="Wheter to use the shrinking heuristic in One Class SVM.",
)
train_parser.add_argument(
    "cache_size",
    type=int,
    default=200,
    help="Specify the size of the kernel cache in MB in One Class SVM.",
)

# Isolation Forest
train_parser.add_argument(
    "n_estimators",
    type=int,
    default=100,
    help="The number of base estimators in the ensemble of Isolation Forest.",
)
train_parser.add_argument(
    "max_features",
    type=float,
    default=1.0,
    help="Number of features to draw from X to train each base estimator in Isolation"
    "Forest",
)
train_parser.add_argument(
    "bootstrap",
    type=bool,
    default=False,
    choices=(True, False),
    help="Indicates if the Bootstrap technique is going to be applied in Isolation"
    " Forest",
)

# Local Outlier Factor
train_parser.add_argument(
    "n_neighbors_lof", type=int, default=20, help="Number of neighbors to use in LOF."
)
train_parser.add_argument(
    "algorithm_lof",
    type=str,
    default="auto",
    choices=("ball_tree", "kd_tree", "brute", "auto"),
    help="Algorithm used to compute the nearest neighbors in LOF.",
)
train_parser.add_argument(
    "leaf_size_lof",
    type=int,
    default=30,
    help="Leaf size passed to BallTree or KDTree in LOF.",
)
train_parser.add_argument(
    "metric_lof",
    type=str,
    default="minkowski",
    help="The distance metric to use for the tree in LOF.",
)
train_parser.add_argument(
    "p_lof", type=int, default=2, help="Parameter of the Minkowski metric in LOF."
)

# k-Nearest Neighbors
train_parser.add_argument(
    "n_neighbors_knn", type=int, default=5, help="Number of neighbors to use in KNN."
)
train_parser.add_argument(
    "radius",
    type=float,
    default=1.0,
    help="The range of parameter space to use by default for radius_neighbors.",
)
train_parser.add_argument(
    "algorithm_knn",
    type=str,
    default="auto",
    choices=("ball_tree", "kd_tree", "brute", "auto"),
    help="Algorithm used to compute the nearest neighbors in KNN.",
)
train_parser.add_argument(
    "leaf_size_knn",
    type=int,
    default=30,
    help="Leaf size passed to BallTree or KDTree in KNN.",
)
train_parser.add_argument(
    "metric_knn",
    type=str,
    default="minkowski",
    help="The distance metric to use for the tree in KNN.",
)
train_parser.add_argument(
    "p_knn", type=int, default=2, help="Parameter of the Minkowski metric in KNN."
)
train_parser.add_argument(
    "score_func",
    type=str,
    default="max_distance",
    choices=("max_distance", "average", "median"),
    help="The function used to score anomalies in KNN.",
)

