import uuid
import datetime
from flask_restx.model import Model
from flask_restx import fields
from flask_mongoengine import Document
from mongoengine.fields import (
    BinaryField,
    StringField,
    ListField,
    DateTimeField,
    BooleanField,
    DictField,
    ReferenceField,
)
from mongoengine import signals, CASCADE
from mongoengine.errors import ValidationError
from blackbox.available_models import AVAILABLE_MODELS


# MongoDB Models
class BlackboxModel(Document):
    """A class which describes the model of a Blackbox inside MongoDB."""

    model_id = StringField(unique=True, required=True)
    creation_date = DateTimeField(default=datetime.datetime.utcnow())
    last_update_date = DateTimeField()
    models = ListField(StringField(), required=True)
    columns = ListField(StringField(), required=True)
    trained = BooleanField(default=False)
    saved = BinaryField()

    meta = {"allow_inheritance": True}

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "creation_date": self.creation_date.isoformat(),
            "last_update_date": self.last_update_date.isoformat(),
            "models": self.models,
            "columns": self.columns,
            "trained": self.trained,
        }

    def clean(self):
        if not all(model in AVAILABLE_MODELS for model in self.models):
            raise ValidationError(
                f"There is at least one model in the list of models that does not "
                f"exist. Passed models: {', '.join(self.models)}. "
                f"Available models: {', '.join(AVAILABLE_MODELS)} "
            )

    @classmethod
    def pre_save(cls, sender, document, **kwargs):
        document.last_update_date = datetime.datetime.utcnow()


signals.pre_save.connect(BlackboxModel.pre_save, sender=BlackboxModel)


class BlackboxPrediction(Document):
    """A class which describes the model of a Blackbox prediction inside MongoDB."""

    prediction_id = StringField(unique=True, required=True, default=str(uuid.uuid4()))
    prediction_date = DateTimeField(default=datetime.datetime.utcnow())
    predictions = DictField(required=True)
    model = ReferenceField(BlackboxModel, reverse_delete_rule=CASCADE, required=True)

    meta = {"allow_inheritance": True}

    def to_dict(self):
        return {
            "prediction_id": self.prediction_id,
            "prediction_date": self.prediction_date,
            "predictions": self.predictions,
        }


# API Models
BlackboxModelApi = Model(
    "BlackboxModel",
    {
        "model_id": fields.String(
            description="Blackbox model id", readonly=True, example="anomaly_model_1"
        ),
        "creation_date": fields.DateTime(
            description="Blackbox model creation date", readonly=True
        ),
        "last_update_date": fields.DateTime(
            description="Blackbox model update date", readonly=True
        ),
        "models": fields.List(
            fields.String(enum=AVAILABLE_MODELS, required=True),
            description="Name of the Machine Learning and Mathematical models inside "
            "the Blackbox model",
            required=True,
            min_items=1,
            example=AVAILABLE_MODELS,
        ),
        "columns": fields.List(
            fields.String,
            description="Name of the columns provided to the Blackbox model",
            required=True,
            min_items=1,
            example=["pressure", "temperature", "humidity"],
        ),
        "trained": fields.Boolean(
            description="Whether the model is already trained or not", readonly=True
        ),
    },
)

BlackboxModelPatchApi = Model(
    "BlackboxModelPatch",
    {
        "models": fields.List(
            fields.String(enum=AVAILABLE_MODELS, required=True),
            description="Name of the Machine Learning and Mathematical models inside "
            "the Blackbox model",
            required=False,
            min_items=1,
            example=AVAILABLE_MODELS,
        ),
        "columns": fields.List(
            fields.String,
            description="Name of the columns provided to the Blackbox model",
            required=False,
            min_items=1,
            example=["pressure", "temperature", "humidity"],
        ),
    },
)

BlackboxPCAMahalanobisApi = Model(
    "BlackboxPCAMahalanobis",
    {
        "n_components": fields.Integer(
            description="Numbers of components for the PCA algorithm",
            default=2,
            example=2,
        )
    },
)

BlackboxAutoencoderApi = Model(
    "BlackboxAutoencoder",
    {
        "hidden_neurons": fields.List(
            fields.Integer,
            description="Neural Network layers and the number of neurons in each layer.",
            min_items=3,
            default=[32, 16, 16, 32],
            example=[32, 16, 16, 32],
        ),
        "dropout_rate": fields.Float(
            description="Dropout rate across all the layers of the Neural Network",
            default=0.2,
            example=0.2,
        ),
        "activation": fields.String(
            description="Layers activation function of Neural Network",
            enum=[
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
            ],
            default="elu",
            example="elu",
        ),
        "kernel_initializer": fields.String(
            description="Layers kernel initializer of Neural Network",
            enum=[
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
            ],
            default="glorot_uniform",
            example="glorot_uniform",
        ),
        "loss_function": fields.String(
            description="Loss function of the Neural Network",
            default="mse",
            example="mse",
        ),
        "optimizer": fields.String(
            description="Optimizer of Neural Network",
            enum=["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"],
            default="adam",
            example="adam",
        ),
        "epochs": fields.Integer(
            description="Number of times that all the batches will be processed in the "
            " Neural Network",
            defualt=100,
            example=100,
        ),
        "batch_size": fields.Integer(description="Batch size", default=32, example=32),
        "validation_split": fields.Float(
            description="Percentage of the training data that will be used for "
            "purpouses in the Neural Network",
            default=0.05,
            example=0.05,
        ),
        "early_stopping": fields.Boolean(
            description="Stops the training process in the Neural Network when it's "
            "not getting any improvement",
            default=False,
            example=False,
        ),
    },
)


BlackboxKMeansApi = Model(
    "BlackboxKMeans",
    {
        "n_clusters": fields.Integer(
            description="Number of clusters for the K-Means algorithm",
            defualt=None,
            example=None,
        ),
        "max_cluster_elbow": fields.Integer(
            description="Maximun number of cluster to test in the Elbow Method",
            default=100,
            example=100,
        ),
    },
)

BlackboxOneClassSVMApi = Model(
    "BlackboxOneClassSVM",
    {
        "kernel": fields.String(
            description="Kernel type for One Class SVM",
            enum=["linear", "poly", "rbf", "sigmoid", "precomputed"],
            default="rbf",
            example="rbf",
        ),
        "degree": fields.Integer(
            description="Degree of the polynomal kernel function for One Class SVM",
            default=3,
            example=3,
        ),
        "gamma": fields.String(
            description="Kernel coefficient for 'rbf', 'poly' and 'sigmoid' in One "
            "Class SVM. It can 'scale', 'auto' or float",
            default="scale",
            example="scale",
        ),
        "coef0": fields.Float(
            description="Independent term in kernel function for One Class SVM. Only "
            "significant in 'poly'",
            default=0.0,
            example=0.0,
        ),
        "tol": fields.Float(
            description="Tolerance for stopping criterion for One Class SVM",
            default=0.001,
            example=0.001,
        ),
        "shrinking": fields.Boolean(
            description="Wheter to use the shrinking heuristic for One Class SVM",
            default=True,
            example=True,
        ),
        "cache_size": fields.Integer(
            description="Specify the size of the kernel cache in MB for One Class SVM",
            default=200,
            example=200,
        ),
    },
)


BlackboxGaussianDistributionApi = Model(
    "BlackboxGaussianDistribution",
    {
        "epsilon_candidates": fields.Integer(
            description="Number of epsilon values that will be tested to find the best one",
            default=100000000,
            example=100000000,
        )
    },
)


BlackboxIsolationForestApi = Model(
    "BlackboxIsolationForest",
    {
        "n_estimators": fields.Integer(
            description="The number of base estimators in the ensemble for Isolation "
            "Forest",
            default=100,
            example=100,
        ),
        "max_features": fields.Float(
            description="Number of features to draw from X to train each base estimator"
            " for Isolation Forest",
            default=1.0,
            example=1.0,
        ),
        "bootstrap": fields.Boolean(
            description="Indicates if the Bootstrap technique is going to be applied "
            "for Isolation FOrest",
            default=False,
            example=False,
        ),
    },
)


BlackboxLOFApi = Model(
    "BlackboxLOF",
    {
        "n_neighbors": fields.Integer(
            description="Number of neighbors to use in LOF", default=20, example=20
        ),
        "algorithm": fields.String(
            description="Algorithm used to compute the nearest neighbors in LOF",
            enum=["ball_tree", "kd_tree", "brute", "auto"],
            default="auto",
            example="auto",
        ),
        "leaf_size": fields.Integer(
            description="Leaf size passed to BallTree or KDTree in LOF",
            default=30,
            example=30,
        ),
        "metric": fields.String(
            description="The distance metric to use for the tree in LOF",
            default="minkowski",
            example="minkowski",
        ),
        "p": fields.Integer(
            description="Paremeter of the Minkowski metric in LOF", default=2, example=2
        ),
    },
)


BlackboxKNNApi = Model(
    "BlackboxKNN",
    {
        "n_neighbors": fields.Integer(
            description="Number of neighbors to use in KNN", default=5, example=5
        ),
        "radius": fields.Float(
            description="The range of parameter space to use by default for "
            "radius_neighbors",
            default=1.0,
            example=1.0,
        ),
        "algorithm": fields.String(
            description="Algorithm used to compute the nearest neighbors in KNN",
            enum=["ball_tree", "kd_tree", "brute", "auto"],
            default="auto",
            example="auto",
        ),
        "leaf_size": fields.Integer(
            description="Leaf size passed to BallTree or KDTree in KNN",
            default=30,
            example=30,
        ),
        "metric": fields.String(
            description="The distance metric to use for the tree in KNN",
            default="minkowski",
            example="minkowski",
        ),
        "p": fields.Integer(
            description="Paremeter of the Minkowski metric in knn", default=2, example=2
        ),
        "score_func": fields.String(
            description="The function used to score anomalies in KNN",
            enum=["max_distance", "average", "median"],
            default="max_distance",
            example="max_distance",
        ),
    },
)


BlackboxDataApi = Model(
    "BlackboxData",
    {
        "columns": fields.List(
            fields.String,
            description="Name of the columns provided to the Blackbox model",
            required=True,
            min_items=1,
            example=["pressure", "temperature", "humidity"],
        ),
        "data": fields.List(
            fields.List(fields.Arbitrary, min_items=1),
            description="List containing the training rows. Each row must have 1 value "
            "for each column indicated in columns ",
            required=True,
            min_items=1,
            example=[
                [60.63, 167.9, 13.64],
                [66.22, 145.3, 14.67],
                [81.76, 143.3, 13.98],
            ],
        ),
    },
)

BlackboxTrainApi = BlackboxDataApi.inherit(
    "BlackboxTrain",
    {
        "contamination": fields.Float(
            description="Contamination fraction in the training dataset",
            default=0.1,
            example=0.1,
        ),
        "n_jobs": fields.Integer(
            description="Number of jobs to use for the computation in algorithms which "
            "supports it",
            default=-1,
            example=-1,
        ),
        "pca_mahalanobis": fields.Nested(BlackboxPCAMahalanobisApi),
        "autoencoder": fields.Nested(BlackboxAutoencoderApi),
        "kmeans": fields.Nested(BlackboxKMeansApi),
        "one_class_svm": fields.Nested(BlackboxOneClassSVMApi),
        "gaussian_distribution": fields.Nested(BlackboxGaussianDistributionApi),
        "isolation_forest": fields.Nested(BlackboxIsolationForestApi),
        "knearest_neighbors": fields.Nested(BlackboxKNNApi),
        "local_outlier_factor": fields.Nested(BlackboxLOFApi),
    },
)


BlackboxResponseApi = Model("BlackboxResponse", {"message": fields.String()})


BlackboxTrainResponseApi = BlackboxResponseApi.inherit(
    "BlackboxTrainResponse", {"task_status": fields.String()}
)


BlackboxResponseErrorApi = Model(
    "BlackboxResponseError",
    {"errors": fields.Wildcard(fields.String), "message": fields.String},
)


BlackboxResponseTaskApi = Model(
    "BlackboxResponseTask",
    {
        "state": fields.String,
        "current": fields.Float,
        "total": fields.Integer,
        "status": fields.String,
    },
)
