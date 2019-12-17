from blackbox.blackbox import BlackBoxAnomalyDetection

# Translation between the API params names and the parameter names of each model
MODEL_PARAMS_NAMES = {
    "PCAMahalanobis": [("n_components", "n_components")],
    "Autoencoder": [
        ("hidden_neurons", "hidden_neurons"),
        ("dropout_rate", "dropout_rate"),
        ("activation", "activation"),
        ("kernel_initializer", "kernel_initializer"),
        ("loss_function", "loss_function"),
        ("optimizer", "optimizer"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("validation_split", "validation_split"),
        ("early_stopping", "early_stopping"),
    ],
    "KMeans": [("n_clusters", "n_clusters")],
    "OneClassSVM": [
        ("kernel", "kernel"),
        ("degree", "degree"),
        ("gamma", "gamma"),
        ("coef0", "coef0"),
        ("tol", "tol"),
        ("shrinking", "shrinking"),
        ("cache_size", "cache_size"),
    ],
    "GaussianDistribution": [],
    "IsolationForest": [
        ("n_estimators", "n_estimators"),
        ("max_features", "max_features"),
        ("bootstrap", "bootstrap"),
    ],
    "LocalOutlierFactor": [
        ("n_neighbors_lof", "n_neighbors"),
        ("algorithm_lof", "algorithm"),
        ("leaf_size_lof", "leaf_size"),
        ("metric_lof", "metric"),
        ("p_lof", "p"),
    ],
    "KNearestNeighbors": [
        ("n_neighbors_knn", "n_neighbors"),
        ("algorithm_knn", "algorithm"),
        ("leaf_size_knn", "leaf_size"),
        ("metric_knn", "metric"),
        ("p_knn", "p"),
    ],
}
