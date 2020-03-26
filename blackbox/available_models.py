# Available models in Blackbox. Have to do this in a separate file because if not then
# the TensorFlow backend will be initialized in modules that is not necessary. 
AVAILABLE_MODELS = [
    "pca_mahalanobis",
    "autoencoder",
    "kmeans",
    "one_class_svm",
    "gaussian_distribution",
    "isolation_forest",
    "knearest_neighbors",
    "local_outlier_factor",
]