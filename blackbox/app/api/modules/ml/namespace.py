from flask import request
from flask_restx import Namespace, Resource
from flask_mongoengine import DoesNotExist, MultipleObjectsReturned, ValidationError
from mongoengine.errors import NotUniqueError
from blackbox.app.celery_app.celery import celery
from blackbox.app.celery_app.tasks_names import CELERY_TRAIN_TASK, CELERY_PREDICT_TASK
from .models import (
    BlackboxModel,
    BlackboxModelApi,
    BlackboxModelPatchApi,
    BlackboxDataApi,
    BlackboxTrainApi,
    BlackboxPCAMahalanobisApi,
    BlackboxAutoencoderApi,
    BlackboxKMeansApi,
    BlackboxOneClassSVMApi,
    BlackboxGaussianDistributionApi,
    BlackboxIsolationForestApi,
    BlackboxKNNApi,
    BlackboxLOFApi,
    BlackboxResponseApi,
    BlackboxTrainResponseApi,
    BlackboxResponseErrorApi,
    BlackboxResponseTaskApi,
)
from .validation import validate_data

# ML Namespace
ml_ns = Namespace(
    name="Blackbox Anomaly Detection",
    path="/api/v1/bb",
    description="Anomaly Detection with Machine Learning and Mathematicals models",
)

# Register models of namespace
ml_ns.add_model("BlackboxModel", BlackboxModelApi)
ml_ns.add_model("BlackboxModelPatch", BlackboxModelPatchApi)
ml_ns.add_model("BlackboxData", BlackboxDataApi)
ml_ns.add_model("BlackboxTrain", BlackboxTrainApi)
ml_ns.add_model("BlackboxPCAMahalanobis", BlackboxPCAMahalanobisApi)
ml_ns.add_model("BlackboxAutoencoder", BlackboxAutoencoderApi)
ml_ns.add_model("BlackboxKMeans", BlackboxKMeansApi)
ml_ns.add_model("BlackboxOneClassSVM", BlackboxOneClassSVMApi)
ml_ns.add_model(
    "BlackboxGaussianDistribution", BlackboxGaussianDistributionApi,
)
ml_ns.add_model("BlackboxIsolationForest", BlackboxIsolationForestApi)
ml_ns.add_model("BlackboxKNN", BlackboxKNNApi)
ml_ns.add_model("BlackboxLOF", BlackboxLOFApi)
ml_ns.add_model("BlackboxResponse", BlackboxResponseApi)
ml_ns.add_model("BlackboxTrainResponse", BlackboxTrainResponseApi)
ml_ns.add_model("BlackboxResponseError", BlackboxResponseErrorApi)
ml_ns.add_model("BlackboxResponseTask", BlackboxResponseTaskApi)


# Error handling
@ml_ns.errorhandler(DoesNotExist)
def handle_does_not_exist(error):
    return {"message": "Blackbox model with the specified id does not exist"}, 404


@ml_ns.errorhandler(NotUniqueError)
def handle_not_unique_error(error):
    return {"message": "Blackbox model with the specified id already exist"}, 409


@ml_ns.errorhandler(MultipleObjectsReturned)
def handle_multiple_objects_returned(error):
    return {"message": "There are several Blackbox models with the specified id"}, 500


@ml_ns.errorhandler(ValidationError)
def handle_validation_error(error):
    return {"message": "Database validation error"}, 400


@ml_ns.route("/models")
class BlackboxModelsMethods(Resource):
    @ml_ns.doc(
        responses={200: "Success"},
        description="Returns all the Blackbox models created",
    )
    @ml_ns.marshal_list_with(BlackboxModelApi)
    def get(self):
        """Returns the list of created Blackbox models"""
        query = BlackboxModel.objects()
        blackbox_models = [model.to_dict() for model in query]
        return blackbox_models, 200


@ml_ns.route("/models/<string:model_id>")
@ml_ns.param("model_id", "Blackbox model id")
class BlackboxModelMethods(Resource):
    @ml_ns.doc(description="Returns the Blackbox model data associated to the id",)
    @ml_ns.response(code=200, description="Success", model=BlackboxModelApi)
    @ml_ns.response(
        code=404,
        description="Blackbox model with specified id does not exist",
        model=BlackboxResponseErrorApi,
    )
    def get(self, model_id):
        """Get a Blackbox model"""
        query = BlackboxModel.objects.get(model_id=model_id)
        return query.to_dict(), 200

    @ml_ns.doc(
        description="Creates a new Blackbox model with the specified models and "
        "expected columns",
    )
    @ml_ns.expect(BlackboxModelApi, validate=True)
    @ml_ns.response(code=200, description="Success", model=BlackboxResponseApi)
    @ml_ns.response(
        code=400,
        description="Payload validation error",
        model=BlackboxResponseErrorApi,
    )
    @ml_ns.response(
        code=409,
        description="Blackbox model with specified id already exist",
        model=BlackboxResponseErrorApi,
    )
    def post(self, model_id):
        """Create a Blackbox model"""
        payload = ml_ns.payload
        newModel = BlackboxModel(
            model_id=model_id, models=payload["models"], columns=payload["columns"],
        )
        newModel.save()
        return (
            {"message": f"Blackbox model with id {model_id} has been created"},
            200,
        )

    @ml_ns.doc(description="Removes a Blackbox model with the specified id")
    @ml_ns.response(code=200, description="Success", model=BlackboxResponseApi)
    @ml_ns.response(
        code=404,
        description="Blackbox model with specified id does not exist",
        model=BlackboxResponseErrorApi,
    )
    def delete(self, model_id):
        """Delete a Blackbox model"""
        query = BlackboxModel.objects.get(model_id=model_id)
        query.delete()
        return (
            {"message": f"Blackbox model with id {model_id} has been deleted"},
            200,
        )

    @ml_ns.doc(description="Update an entire Blackbox model")
    @ml_ns.expect(BlackboxModelApi, validate=True)
    @ml_ns.response(code=200, description="Success", model=BlackboxResponseApi)
    @ml_ns.response(
        code=400,
        description="Payload validation error",
        model=BlackboxResponseErrorApi,
    )
    @ml_ns.response(
        code=404,
        description="Blackbox model with specified id does not exist",
        model=BlackboxResponseErrorApi,
    )
    def put(self, model_id):
        """Update an entire Blackbox model"""
        payload = ml_ns.payload
        blackbox_model = BlackboxModel.objects.get(model_id=model_id)
        blackbox_model.modify(
            models=payload["models"], columns=payload["columns"],
        )
        # Necesary to execute the pre_save signal and update the update date
        blackbox_model.save()
        return (
            {"message": f"Blackbox model with id {model_id} has been updated"},
            200,
        )

    @ml_ns.doc(description="Update a Blackbox model")
    @ml_ns.expect(BlackboxModelApi, validate=True)
    @ml_ns.response(code=200, description="Success", model=BlackboxResponseApi)
    @ml_ns.response(
        code=400,
        description="Payload validation error",
        model=BlackboxResponseErrorApi,
    )
    @ml_ns.response(
        code=404,
        description="Blackbox model with specified id does not exist",
        model=BlackboxResponseErrorApi,
    )
    def patch(self, model_id):
        """Update a Blackbox model"""
        payload = ml_ns.payload
        blackbox_model = BlackboxModel.objects.get(model_id=model_id)

        if payload.get("models", None):
            blackbox_model.models = payload["models"]

        if payload.get("columns", None):
            blackbox_model.columns = payload["columns"]

        blackbox_model.save()

        return {"message": f"Blackbox model with id {model_id} has been updated"}, 200


@ml_ns.route("/models/<string:model_id>/train")
@ml_ns.param("model_id", "Blackbox model id")
class TrainMethods(Resource):
    @ml_ns.expect(BlackboxTrainApi, validate=True)
    @ml_ns.response(code=202, model=BlackboxTrainResponseApi, description="Success")
    @ml_ns.response(
        code=400, model=BlackboxResponseErrorApi, description="Payload validation error"
    )
    @ml_ns.response(
        code=404,
        model=BlackboxResponseErrorApi,
        description="Blackbox model with specified id does not exist",
    )
    def post(self, model_id):
        """Train a Blackbox model"""
        payload = ml_ns.payload
        blackbox_model = BlackboxModel.objects.get(model_id=model_id)

        # Validate training data
        errors = validate_data(
            payload["columns"], payload["data"], blackbox_model.columns
        )

        if errors:
            return {"errors": errors, "message": "Input payload validation error"}, 400

        # Training data
        data = {"columns": payload["columns"], "data": payload["data"]}

        # Models parameters
        models_parameters = {}
        for model in blackbox_model.models:
            try:
                models_parameters[model] = payload[model]
            except KeyError:
                models_parameters[model] = {}

        # Create training task
        task = celery.send_task(
            CELERY_TRAIN_TASK, args=[model_id, data, models_parameters],
        )

        return (
            (
                {
                    "message": "A task to train the model has been started",
                    "task_status": f"http://{request.host}/{ml_ns.path}/task/{task.id}",
                }
            ),
            202,
        )


@ml_ns.route("/models/<string:model_id>/predict")
@ml_ns.param("model_id", "Blackbox model id")
class PredictMethods(Resource):
    @ml_ns.expect(BlackboxDataApi)
    @ml_ns.response(code=202, model=BlackboxTrainResponseApi, description="Success")
    @ml_ns.response(
        code=400, model=BlackboxResponseErrorApi, description="Payload validation error"
    )
    @ml_ns.response(
        code=404,
        model=BlackboxResponseErrorApi,
        description="Blackbox model with specified id does not exist",
    )
    def post(self, model_id):
        """Predict with a Blackbox model"""
        payload = ml_ns.payload
        blackbox_model = BlackboxModel.objects.get(model_id=model_id)

        # Validate training data
        errors = validate_data(
            payload["columns"], payload["data"], blackbox_model.columns
        )

        if errors:
            return {"errors": errors, "message": "Input payload validation error"}, 400

        # Data to predict
        data = {"columns": payload["columns"], "data": payload["data"]}

        # Create predicting task
        task = celery.send_task(CELERY_PREDICT_TASK, args=[model_id, data])

        return (
            (
                {
                    "message": "A task to predict anomalies with Blackbox model with "
                    f"id {model_id} has been started",
                    "task_status": f"http://{request.host}/{ml_ns.path}/task/{task.id}",
                }
            ),
            202,
        )


@ml_ns.route("/task/<string:task_id>")
@ml_ns.param("task_id", "Celery task id")
class TaskStatus(Resource):
    @ml_ns.doc(
        description="Returns the progress of the task specifying the state, the current"
        " progress, the total progress that has to be reached and the status. Also, if"
        " the task has ended a result will be returned too.",
    )
    @ml_ns.response(code=200, description="Success", model=BlackboxResponseTaskApi)
    def get(self, task_id):
        """Get the status of a task"""
        task = celery.AsyncResult(task_id)
        if task.state == "PENDING":
            response = {
                "state": task.state,
                "current": 0,
                "total": 100,
                "status": "Pending...",
            }
        elif task.state != "FAILURE":
            response = {
                "state": task.state,
                "current": task.info.get("current", 0),
                "total": task.info.get("total", 100),
                "status": task.info.get("status", ""),
            }
            if "result" in task.info:
                response["result"] = task.info["result"]
        else:
            response = {
                "state": task.state,
                "current": 100,
                "total": 100,
                "status": str(task.info),
            }

        return response, 200
