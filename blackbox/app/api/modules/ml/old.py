import os
import threading
from datetime import datetime
from dateutil import parser
from flask import request
from flask_restx import Resource, Namespace, cors
from werkzeug.utils import secure_filename
from blackbox import settings
from blackbox.utils.api import (
    read_json,
    add_entity_json,
    build_url,
    update_entity_json,
    delete_entity_json,
    match_regex,
    parse_float,
)
from .parsers import train_parser
from .models import (
    new_entity_model,
    update_entity,
    update_entity_model,
    update_entity_models,
)
from .models_params import MODEL_PARAMS_NAMES
from blackbox.utils.worker import celery_app
from blackbox.blackbox import BlackBoxAnomalyDetection
from blackbox.models import unsupervised

# Lock
lock = threading.Lock()

# ML Namespace
ml_ns = Namespace(
    settings.API_ANOMALY_ENDPOINT, description="Anomaly Detection with Machine Learning"
)

# Add models of namespace
ml_ns.add_model("new_entity", new_entity_model)
ml_ns.add_model("model", update_entity_model)
ml_ns.add_model("models", update_entity_models)
ml_ns.add_model("update_entity", update_entity)

# API Routes
@ml_ns.route("/available_models")
class AvailableModels(Resource):
    @cors.crossdomain(origin="*")
    @ml_ns.doc(
        responses={200: "Success"},
        description="Return the list of available Machine Learning models for Anomaly "
        "Detection",
    )
    def get(self):
        """Returns ML available models"""
        return {"available_models": BlackBoxAnomalyDetection.AVAILABLE_MODELS}, 200


@ml_ns.route("/available_models/<string:model_name>")
@ml_ns.param("model_name", "Name of anomaly detection model")
class AnomalyModel(Resource):
    @cors.crossdomain(origin="*")
    @ml_ns.doc(
        responses={200: "Success", 400: "Model does not exist"},
        description="Return the description of the model.",
    )
    def get(self, model_name):
        """Returml_ns model description"""
        if model_name not in BlackBoxAnomalyDetection.AVAILABLE_MODELS:
            return {"error": "Model does not exist."}, 400

        description = ""

        if model_name == "PCAMahalanobis":
            description = unsupervised.AnomalyPCAMahalanobis.__doc__

        if model_name == "Autoencoder":
            description = unsupervised.AnomalyAutoencoder.__doc__

        if model_name == "KMeaml_ns":
            description = unsupervised.AnomalyKMeaml_ns.__doc__

        if model_name == "OneClassSVM":
            description = unsupervised.AnomalyOneClassSVM.__doc__

        if model_name == "GaussianDistribution":
            description = unsupervised.AnomalyGaussianDistribution.__doc__

        if model_name == "IsolationForest":
            description = unsupervised.AnomalyIsolationForest.__doc__

        if model_name == "KNearestNeighbors":
            description = unsupervised.AnomalyKNN.__doc__

        if model_name == "LocalOutlierFactor":
            description = unsupervised.AnomalyLOF.__doc__

        return {"model": model_name, "description": description}, 200


@ml_ns.route("/models")
class ModelsList(Resource):
    @cors.crossdomain(origin="*")
    @ml_ns.doc(
        responses={200: "Success"}, description="Return the list of created models",
    )
    def get(self):
        """Returns the list of Blackbox models"""
        json_models = read_json(settings.MODELS_ROUTE_JSON)
        if not json_models:
            json_models = {}

        return json_models, 200


@ml_ns.route("/models/<string:model_id>")
@ml_ns.param("model_id", "Blackbox model id")
class Model(Resource):
    @cors.crossdomain(origin="*")
    @ml_ns.doc(
        responses={200: "Success", 400: "Model or JSON file does not exist"},
        description="Returml_ns a model and its training attributes",
    )
    def get(self, model_id):
        """Return an entity"""
        json_models = read_json(settings.MODELS_ROUTE_JSON)
        if not json_models:
            return {"error": "The JSON file does not exist"}, 400

        entity_regex = match_regex(list(json_models.keys()), model_id)
        if not entity_regex:
            return (
                {"error": "The model does not match any model_id in model.json"},
                400,
            )

        entity_data = json_models[entity_regex]
        return {model_id: entity_data}, 200

    @cors.crossdomain(origin="*")
    @ml_ns.doc(
        body=new_entity_model,
        responses={
            200: "Success",
            400: "No payload, unable to create the model or validation error",
        },
        description="Creates a new Blackbox model with model_id.",
    )
    def post(self, model_id):
        """Creates a Blackbox model"""
        if not request.json:
            return {"error": "No payload was send"}, 400

        try:
            attrs = request.json["attrs"]

            if not isinstance(attrs, list):
                return {"error": "attrs has to be a list with strings iml_nside"}, 400
        except KeyError:
            return {"error": "No payload with attrs was send"}, 400
        except TypeError:
            return {"error": "No payload with attrs was send"}, 400

        with lock:
            created, msg = add_entity_json(
                settings.MODELS_ROUTE_JSON,
                model_id,
                os.path.join(settings.MODELS_ROUTE, model_id),
                attrs,
            )

        if not created:
            return {"error": msg}, 400

        return {"message": msg}, 200

    @cors.crossdomain(origin="*")
    @ml_ns.doc(
        body=update_entity,
        responses={
            200: "Success",
            400: "No payload, unable to write or validation error",
        },
        description="Updates an entity with the specified model_id. The new values of"
        " the entity has to be specified in the payload. It's mandatory to specify"
        " every value. If the new_model_id is specified, it must not exist already."
        " If the default model is updated it must exist in the list of trained model"
        " for the specified entity.",
    )
    def put(self, model_id):
        """Updates an entity"""
        if not request.json:
            return {"error": "No payload was sent"}, 400

        json_ = request.json
        new_model_id = None
        default = None
        attrs = None
        new_models = None

        if "new_model_id" in json_:
            new_model_id = json_["new_model_id"]

        if "default" in json_:
            default = json_["default"]

        if "attrs" in json_:
            attrs = json_["attrs"]

        if "models" in json_:
            new_models = json_["models"]

        with lock:
            updated, messages = update_entity_json(
                model_id,
                settings.MODELS_ROUTE_JSON,
                settings.MODELS_ROUTE,
                new_model_id,
                default,
                attrs,
                new_models,
            )

        if not updated:
            return {"error": "The entity was not updated", "messages": messages}, 400

        return {"messages": messages}, 200

    @cors.crossdomain(origin="*")
    @ml_ns.doc(
        responses={200: "Success", 400: "Entity or JSON file does not exist"},
        description="Deletes an entity from the API list and moves its trained models"
        " and training data to the trash directory from the API.",
    )
    def delete(self, model_id):
        """Deletes an entity"""
        with lock:
            deleted, msg = delete_entity_json(
                model_id,
                settings.MODELS_ROUTE_JSON,
                settings.MODELS_ROUTE,
                settings.MODELS_ROUTE_TRASH,
            )

        if not deleted:
            return {"error": msg}, 400

        return {"message": msg}, 200


@ml_ns.route("/<string:model_id>/train")
@ml_ns.param("model_id", "Orion Context Broker (FIWARE) entity ID")
class Train(Resource):
    @cors.crossdomain(origin="*")
    @ml_ns.expect(train_parser)
    @ml_ns.doc(
        responses={
            200: "Success",
            400: "Entity or JSON file does not exist, no training file provided or"
            " input arguments were not specified",
        },
        description="Train a Blackbox Anomaly Detection Model for an entity with the"
        " specified model_id. The entity has to be already created. The Blackbox Model"
        " will be trained with the uploaded file. The process of training will be"
        " asynchronous and an URL will be returned in order to see the training"
        " progress.",
    )
    def post(self, model_id):
        """Traiml_ns a Blackbox model"""
        json_models = read_json(settings.MODELS_ROUTE_JSON)
        if not json_models or model_id not in json_models:
            return {"error": "The entity does not exist!"}, 400

        parsed_args = train_parser.parse_args()

        if not parsed_args.get("file"):
            return (
                {
                    "error": "No file was provided to train the model for the entity {}".format(
                        model_id
                    )
                },
                400,
            )

        if parsed_args.get("input_arguments"):
            input_arguments = parsed_args.get("input_arguments").split(",")
        else:
            return {"error": "Input arguments were not specified for the model"}, 400

        if parsed_args.get("name"):
            model_name = parsed_args.get("name")
        else:
            date = datetime.now()
            model_name = "model_{}_{}-{}-{}-{}:{}".format(
                model_id, date.year, date.month, date.day, date.hour, date.minute
            )

        if parsed_args.get("models"):
            models = parsed_args.get("models").split(",")
        else:
            models = BlackBoxAnomalyDetection.AVAILABLE_MODELS

        # Get additonal models params
        additional_params = {
            "PCAMahalanobis": {},
            "Autoencoder": {},
            "KMeaml_ns": {},
            "OneClassSVM": {},
            "GaussianDistribution": {},
            "IsolationForest": {},
            "KNearestNeighbors": {},
            "LocalOutlierFactor": {},
        }

        for anomaly_model_name in MODEL_PARAMS_NAMES.keys():
            for param in MODEL_PARAMS_NAMES[anomaly_model_name]:
                # Traml_nslation tuple from param API name to model param name
                param_api_name, param_model_real_name = param
                param_value = parsed_args.get(param_api_name)
                if param_value:
                    additional_params[anomaly_model_name][
                        param_model_real_name
                    ] = param_value

        # Save the file
        file = request.files["file"]
        _, ext = os.path.splitext(file.filename)
        if ext != ".csv":
            return {"error": "The file is not a .csv file"}, 400

        file.save(
            os.path.join(
                settings.MODELS_ROUTE,
                model_id,
                "train_data",
                secure_filename(file.filename),
            )
        )

        # Train the model
        path_train_file = (
            settings.MODELS_ROUTE + "/" + model_id + "/train_data/" + file.filename
        )
        task = celery_app.send_task(
            "tasks.train",
            args=[
                model_id,
                path_train_file,
                model_name,
                models,
                input_arguments,
                additional_params,
            ],
        )

        return (
            {
                "message": "The file {} was uploaded. Training model for entity {}".format(
                    file.filename, model_id
                ),
                "task_status_url": build_url(
                    request.url_root, settings.API_ANOMALY_ENDPOINT, "task", task.id
                ),
            },
            202,
        )


@ml_ns.route("/ocb_predict")
class OCBPredict(Resource):
    @cors.crossdomain(origin="*")
    @ml_ns.doc(
        responses={
            202: "Success",
            400: "No payload, the entity or the JSON file does not exist or an attr is"
            " missing",
        },
        description="This endpoint will receive the data from Orion Context Broker"
        " (FIWARE), i.e this endpoint has to be specified in the HTTP URL field of a"
        " OCB subscription (OCB will make a POST to this endpoint). With the data"
        " received from an entity, a prediction will be made using the default"
        " pre-trained model for the entity.",
    )
    def post(self):
        """Endpoint to receive data from OCB and predict if it's an anomaly."""
        json_models = read_json(settings.MODELS_ROUTE_JSON)
        if not json_models:
            return (
                {
                    "error": "The file {} does not exist".format(
                        settings.MODELS_ROUTE_JSON
                    )
                },
                400,
            )

        if not request.json:
            return {"error": "No payload in request"}, 400

        data = request.json["data"][0]
        model_id = data["id"]

        entity_regex = match_regex(list(json_models.keys()), model_id)
        if not entity_regex:
            return (
                {"error": "The entity does not match any entity name in model.json"},
                400,
            )

        entity = json_models[entity_regex]
        if (entity["default"] is None and len(entity["models"]) == 0) or (
            len(entity["models"]) == 0
        ):
            return {"error": "The entity has not trained models"}, 400

        default = entity["default"]
        if default is None:
            # if the default model is not set, take the first model from the dict of models
            entity_models = entity["models"]
            model = list(entity_models.keys())[0]
        else:
            model = entity["models"][default]

        predict_data = []
        for attr in model["input_arguments"]:
            try:
                date = data[attr]["metadata"]["dateModified"]["value"]
                predict_data.append(data[attr]["value"])
            except KeyError:
                return (
                    {"error": "The attr {} was not in the sent attrs".format(attr)},
                    400,
                )

        # parse strings to float
        predict_data = parse_float(predict_data)

        # parse date
        date = parser.parse(date).strftime("%Y-%m-%d %H:%M:%S")
        model_path = model["model_path"]
        task = celery_app.send_task(
            "tasks.predict", args=[model_id, date, model_path, predict_data]
        )

        return (
            {
                "message": "The prediction for {} is being made...",
                "task_status_url": build_url(
                    request.url_root, settings.API_ANOMALY_ENDPOINT, "task", task.id
                ),
            },
            202,
        )


@ml_ns.route("/predict")
class Predict(Resource):
    @cors.crossdomain(origin="*")
    def post(self):
        """Endpoint to receive data from an entity and predict if it's an anomaly"""
        pass


@ml_ns.route("/task/<string:task_id>")
@ml_ns.param("task_id", "Celery task id")
class TaskStatus(Resource):
    @cors.crossdomain(origin="*")
    @ml_ns.doc(
        responses={200: "Success"},
        description="Returml_ns the progress of the task specifying the state, the current"
        " progress, the total progress that has to be reached and the status. Also, if"
        " the task has ended a result will be returned too.",
    )
    def get(self, task_id):
        """Gets the status of a task"""
        task = celery_app.AsyncResult(task_id)
        if task.state == "PENDING":
            respoml_nse = {
                "state": task.state,
                "current": 0,
                "total": 100,
                "status": "Pending...",
            }
        elif task.state != "FAILURE":
            respoml_nse = {
                "state": task.state,
                "current": task.info.get("current", 0),
                "total": task.info.get("total", 100),
                "status": task.info.get("status", ""),
            }
            if "result" in task.info:
                respoml_nse["result"] = task.info["result"]
        else:
            respoml_nse = {
                "state": task.state,
                "current": 100,
                "total": 100,
                "status": str(task.info),
            }

        return respoml_nse, 200


@ml_ns.errorhandler
def handle_root_exception(error):
    """Namespace error handler"""
    return {"error": str(error)}, getattr(error, "code", 500)
