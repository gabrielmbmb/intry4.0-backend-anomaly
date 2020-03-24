import datetime
from flask_restx.model import Model
from flask_restx import fields
from flask_mongoengine import Document
from mongoengine.fields import FileField, StringField, ListField, DateTimeField
from mongoengine import signals
from mongoengine.errors import ValidationError
from blackbox.blackbox import AVAILABLE_MODELS


# MongoDB Models
class BlackboxModel(Document):
    model_id = StringField(unique=True, required=True)
    creation_date = DateTimeField(default=datetime.datetime.utcnow())
    last_update_date = DateTimeField()
    models = ListField(StringField(), required=True)
    columns = ListField(StringField(), required=True)
    model_file = FileField()

    def to_dict(self):
        return {
            "model_id": self.model_id,
            "creation_date": self.creation_date.isoformat(),
            "last_update_date": self.last_update_date.isoformat(),
            "models": self.models,
            "columns": self.columns,
        }

    def clean(self):
        print("cleaning")
        if not all(model in AVAILABLE_MODELS for model in self.models):
            print("yup")
            raise ValidationError(
                f"There is at least one model in the list of models that does not "
                f"exist. Passed models: {', '.join(self.models)}. "
                f"Available models: {', '.join(AVAILABLE_MODELS)} "
            )

    @classmethod
    def pre_save(cls, sender, document, **kwargs):
        document.last_update_date = datetime.datetime.utcnow()


signals.pre_save.connect(BlackboxModel.pre_save, sender=BlackboxModel)

# API Models
BlackboxModelApi = Model(
    "BlackboxModel",
    {
        "model_id": fields.String(readonly=True, example="anomaly_model_1"),
        "creation_date": fields.DateTime(readonly=True),
        "last_update_date": fields.DateTime(readonly=True),
        "models": fields.List(
            fields.String(enum=AVAILABLE_MODELS, required=True),
            description="Name of the Machine Learning and Mathematical models inside "
            "the Blackbox model",
            required=True,
            example=AVAILABLE_MODELS,
        ),
        "columns": fields.List(
            fields.String,
            description="Name of the columns provided to the Blackbox model",
            required=True,
            example=["pressure", "temperature", "humidity"],
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
            example=AVAILABLE_MODELS,
        ),
        "columns": fields.List(
            fields.String,
            description="Name of the columns provided to the Blackbox model",
            required=False,
            example=["pressure", "temperature", "humidity"],
        ),
    },
)

BlackboxPCAMahalanobisApi = Model(
    "BlackboxPCAMahalanobis",
    {
        "n_components": fields.Integer(
            description="Numbers of components for the PCA algorithm", default=2
        )
    },
)

BlackboxAutoencoderApi = Model("BlackboxAutoencoder", {})
BlackboxKMeansApi = Model("BlackboxKMeans", {})
BlackboxOneClassSVMApi = Model("BlackboxOneClassSVM", {})
BlackboxGaussianDistributionApi = Model("BlackboxGaussianDistribution", {})
BlackboxIsolationForestApi = Model("BlackboxIsolationForest", {})
BlackboxKNNApi = Model("BlackboxKNN", {})
BlackboxLOFApi = Model("BlackboxLOF", {})


BlackboxDataApi = Model(
    "BlackboxData",
    {
        "columns": fields.List(
            fields.String,
            description="Name of the columns provided to the Blackbox model",
            required=True,
            example=["pressure", "temperature", "humidity"],
        ),
        "data": fields.List(
            fields.List(fields.Arbitrary),
            description="List containing the training rows. Each row must have 1 value "
            "for each column indicated in columns ",
            required=True,
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
        "PCAMahalanobis": fields.Nested(BlackboxPCAMahalanobisApi),
        "Autoencoder": fields.Nested(BlackboxAutoencoderApi),
        "KMeans": fields.Nested(BlackboxKMeansApi),
        "OneClassSVM": fields.Nested(BlackboxOneClassSVMApi),
        "GaussianDistribution": fields.Nested(BlackboxGaussianDistributionApi),
        "IsolationForest": fields.Nested(BlackboxIsolationForestApi),
        "KNearestNeighbors": fields.Nested(BlackboxKNNApi),
        "LocalOutlierFactor": fields.Nested(BlackboxLOFApi),
    },
)
