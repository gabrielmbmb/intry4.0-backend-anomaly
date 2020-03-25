from flask import request
from flask_restx import Namespace, Resource
from flask_mongoengine import DoesNotExist, MultipleObjectsReturned, ValidationError
from mongoengine.errors import NotUniqueError
from blackbox.app.celery_app.celery import celery
from blackbox.app.celery_app.tasks_names import CELERY_TRAIN_TASK
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
)


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
    @ml_ns.doc(responses={200: "Success"})
    @ml_ns.marshal_list_with(BlackboxModelApi)
    def get(self):
        """Returns the list of created Blackbox models"""
        query = BlackboxModel.objects()
        blackbox_models = [model.to_dict() for model in query]
        return blackbox_models, 200


@ml_ns.route("/models/<string:model_id>")
@ml_ns.param("model_id", "Blackbox model id")
class BlackboxModelMethods(Resource):
    @ml_ns.doc(
        responses={
            200: "Success",
            404: "Blackbox model with specified id does not exist",
            500: "Server internal error",
        },
        description="Returns the Blackbox model data associated to the id.",
    )
    @ml_ns.marshal_with(BlackboxModelApi)
    def get(self, model_id):
        """Get a Blackbox model"""
        query = BlackboxModel.objects.get(model_id=model_id)
        return query.to_dict(), 200

    @ml_ns.doc(
        body=BlackboxModelApi,
        responses={
            200: "Success",
            400: "Payload validation error",
            409: "Blackbox model with specified id already exist",
        },
        description="Creates a new Blackbox model",
    )
    @ml_ns.expect(BlackboxModelApi, validate=True)
    @ml_ns.marshal_with(BlackboxResponseApi)
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

    @ml_ns.doc(
        responses={
            200: "Success",
            404: "Blackbox model with specified id does not exist",
            500: "Could not remove the Blackbox model with the specified id",
        },
    )
    @ml_ns.marshal_with(BlackboxResponseApi)
    def delete(self, model_id):
        """Delete a Blackbox model"""
        query = BlackboxModel.objects.get(model_id=model_id)
        query.delete()
        return (
            {"message": f"Blackbox model with id {model_id} has been deleted"},
            200,
        )

    @ml_ns.doc(
        body=BlackboxModelApi,
        responses={
            200: "Success",
            400: "Payload validation error",
            404: "Blackbox model with specified id does not exist",
        },
    )
    @ml_ns.expect(BlackboxModelApi, validate=True)
    @ml_ns.marshal_with(BlackboxResponseApi)
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

    @ml_ns.doc(
        body=BlackboxModelPatchApi,
        responses={
            200: "Success",
            400: "Payload validation error",
            404: "Blackbox model with specified id does not exist",
        },
    )
    @ml_ns.expect(BlackboxModelPatchApi, validate=True)
    @ml_ns.marshal_with(BlackboxResponseApi)
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
    @ml_ns.doc(
        body=BlackboxTrainApi,
        responses={
            202: "Success",
            400: "Payload validation error",
            404: "Blackbox model with specified id does not exist",
        },
    )
    @ml_ns.expect(BlackboxTrainApi, validate=True)
    @ml_ns.marshal_with(BlackboxTrainResponseApi, code=202)
    def post(self, model_id):
        """Train a Blackbox model"""
        payload = ml_ns.payload
        blackbox_model = BlackboxModel.objects.get(model_id=model_id)

        # Check that columns provided are the same as the ones in the model
        if not all(column in blackbox_model.columns for column in payload["columns"]):
            return (
                {
                    "errors": {
                        "columns": "the provided columns are not the same as those "
                        "previously created in the Blackbox model"
                    },
                    "message": "Input payload validation error",
                },
                400,
            )

        # Check that data list is not empty
        if not payload.get("data", None):
            return (
                {
                    "errors": {"data": "data cannot be empty"},
                    "message": "Input payload validation failed",
                },
                400,
            )

        # Check if every row in training data has the same length
        rows_lengths = set(list(map(len, payload["data"])))
        if len(rows_lengths) != 1:
            return (
                {
                    "errors": {"data": "the rows have to be the same length"},
                    "message": "Input payload validation error",
                },
                400,
            )

        # Check if the rows length is equal to the columns length
        columns_length = len(payload["columns"])
        rows_length = rows_lengths.pop()
        if rows_length != columns_length:
            return (
                {
                    "errors": {
                        "data": f"the rows have {rows_length} elements but "
                        f"{columns_length} columns were provided"
                    },
                    "message": "Input payload validation error",
                },
                400,
            )

        # Create training task
        task = celery.send_task(
            CELERY_TRAIN_TASK, args=[payload["columns"], payload["data"]],
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
    @ml_ns.doc(
        body=BlackboxDataApi,
        responses={
            202: "Success",
            400: "Payload validation error",
            404: "Blackbox model with specified id does not exist",
        },
    )
    @ml_ns.expect(BlackboxDataApi)
    def post(self, model_id):
        """Predict with a Blackbox model"""
        pass


@ml_ns.route("/task/<string:task_id>")
@ml_ns.param("task_id", "Celery task id")
class TaskStatus(Resource):
    @ml_ns.doc(
        responses={200: "Success"},
        description="Returns the progress of the task specifying the state, the current"
        " progress, the total progress that has to be reached and the status. Also, if"
        " the task has ended a result will be returned too.",
    )
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
