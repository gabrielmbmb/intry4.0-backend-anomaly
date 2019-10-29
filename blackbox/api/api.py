import os
from blackbox import settings
from dateutil import parser
from datetime import datetime
from flask import Flask, request
from flask_restplus import Api, Resource, cors, fields
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from blackbox.api.api_utils import read_json, add_entity_json, build_url, update_entity_json, \
    delete_entity_json
from blackbox.api.worker import celery_app
from blackbox.blackbox import BlackBoxAnomalyDetection

# Todo: add logging to the Flask API

# Create Flask App
app = Flask(settings.APP_NAME)
api = Api(app, version='0.0.1', title=settings.APP_NAME, description=settings.APP_DESC, doc='/swagger')

# API Namespaces
anomaly_ns = api.namespace(settings.API_ANOMALY_ENDPOINT, description='Anomaly Detection Operations')

# API parsers
file_parser = anomaly_ns.parser()
file_parser.add_argument('file', type=FileStorage, required=True, location='files', help='CSV training file')

# API Models
new_entity_model = anomaly_ns.model('new_entity', {
    'attrs': fields.List(fields.String(), description='New entity attributes expected to train the models.',
                         required=True)
})

update_entity_model = anomaly_ns.model('model', {
    'model_path': fields.String(description='Path of the model', required=False),
    'train_data_path': fields.String(description='Path of the training data', required=False),
})

update_entity_models = anomaly_ns.model('models', {
    'model': fields.Nested(update_entity_model)
})

update_entity = anomaly_ns.model('update_entity', {
    'new_entity_id': fields.String(description='New entity id', required=False),
    'default': fields.String(description='New default model', required=False),
    'attrs': fields.List(fields.String(), description='New entity attributes', required=False),
    'models': fields.Nested(update_entity_model)
})


# API Routes
@anomaly_ns.route('/models')
class AvailableModels(Resource):
    @anomaly_ns.doc(responses={200: 'Success'},
                    description='Returns the list of available models for Anomaly Detection')
    def get(self):
        """Returns available models"""
        return {
                   'available_models': BlackBoxAnomalyDetection.AVAILABLE_MODELS
               }, 200


@anomaly_ns.route('/models/<string:model_name>')
@anomaly_ns.param('model_name', 'Name of anomaly detection model')
class AnomalyModel(Resource):
    @anomaly_ns.doc(responses={200: 'Success', 400: 'Model does not exist'},
                    description='Return the description of the model.')
    def get(self, model_name):
        """Returns model description"""
        if model_name not in BlackBoxAnomalyDetection.AVAILABLE_MODELS:
            return {
                       'error': 'Model does not exist.'
                   }, 400

        return {
                   'model': model_name,
                   'description': 'Not implemented yet!'
               }, 200


@anomaly_ns.route('/entities')
class EntitiesList(Resource):
    @cors.crossdomain(origin='*')
    @anomaly_ns.doc(responses={200: 'Success'},
                    description="Return the list of created entities and for each entity its attributes, "
                                "default model and trained models.")
    def get(self):
        """Returns a list of entities"""
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if not json_entities:
            json_entities = {}

        return json_entities, 200


@anomaly_ns.route('/entities/<string:entity_id>')
@anomaly_ns.param('entity_id', 'Orion Context Broker (FIWARE) entity ID')
class Entity(Resource):
    @cors.crossdomain(origin='*')
    @anomaly_ns.doc(responses={200: 'Success', 400: 'Entity or JSON file does not exist'},
                    description='Returns an entity and its attributes, default model and trained models.')
    def get(self, entity_id):
        """Return an entity"""
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if not json_entities:
            return {'error': 'The JSON file does not exist'}, 400

        if entity_id not in json_entities:
            return {'error': 'The entity {} does not exist'.format(entity_id)}, 400

        entity_data = json_entities[entity_id]
        return {entity_id: entity_data}, 200

    @cors.crossdomain(origin='*')
    @anomaly_ns.doc(body=new_entity_model,
                    responses={200: 'Success', 400: 'No payload, unable to create the entity or validation error'},
                    description="Creates an entity with the specified entity_id which has to be the same as in Orion "
                                "Context Broker (OCB, FIWARE). It is necessary to specify the name of the attributes "
                                "which the API will receive from OCB in order to make the predictions. The number of "
                                "the attributes has to be exactly the same as the number of attributes in the training "
                                "dataset.")
    def post(self, entity_id):
        """Creates an entity"""
        if not request.json:
            return {'error': 'No payload was send'}, 400

        try:
            attrs = request.json['attrs']

            if not isinstance(attrs, list):
                return {'error': 'attrs has to be a list with strings inside'}, 400
        except KeyError:
            return {'error': 'No payload with attrs was send'}, 400
        except TypeError:
            return {'error': 'No payload with attrs was send'}, 400

        created, msg = add_entity_json(settings.MODELS_ROUTE_JSON, entity_id,
                                       os.path.join(settings.MODELS_ROUTE, entity_id), attrs)
        if not created:
            return {'error': msg}, 400

        return {'message': msg}, 200

    @cors.crossdomain(origin='*')
    @anomaly_ns.doc(body=update_entity,
                    responses={200: 'Success', 400: 'No payload, unable to write or validation error'},
                    description="Updates an entity with the specified entity_id. The new values of the entity has to "
                                "be specified in the payload. It's mandatory to specify every value. If the "
                                "new_entity_id is specified, it must not exist already. If the default model is "
                                "updated it must exist in the list of trained model for the specified entity.")
    def put(self, entity_id):
        """Updates an entity"""
        if not request.json:
            return {
                       'error': 'No payload was sent'
                   }, 400

        json_ = request.json
        new_entity_id = None
        default = None
        attrs = None
        models = None

        if 'new_entity_id' in json_:
            new_entity_id = json_['new_entity_id']

        if 'default' in json_:
            default = json_['default']

        if 'attrs' in json_:
            attrs = json_['attrs']

        if 'models' in json_:
            models = json_['models']

        updated, messages = update_entity_json(entity_id, settings.MODELS_ROUTE_JSON, settings.MODELS_ROUTE,
                                               new_entity_id, default, attrs, models)

        if not updated:
            return {
                       'error': 'The entity was not updated',
                       'messages': messages
                   }, 400

        return {
                   'messages': messages
               }, 200

    @cors.crossdomain(origin='*')
    @anomaly_ns.doc(responses={200: 'Success', 400: 'Entity or JSON file does not exist'},
                    description="Deletes an entity from the API list and moves its trained models and training data "
                                "to the trash directory from the API.")
    def delete(self, entity_id):
        """Deletes an entity"""
        deleted, msg = delete_entity_json(entity_id, settings.MODELS_ROUTE_JSON, settings.MODELS_ROUTE,
                                          settings.MODELS_ROUTE_TRASH)

        if not deleted:
            return {
                       'error': msg
                   }, 400

        return {'message': msg}, 200


@anomaly_ns.route('/train/<string:entity_id>')
@anomaly_ns.param('entity_id', 'Orion Context Broker (FIWARE) entity ID')
class Train(Resource):
    @cors.crossdomain(origin='*')
    @anomaly_ns.expect(train_parser)
    @anomaly_ns.doc(responses={
        200: 'Success',
        400: 'Entity or JSON file does not exist, no training file provided or input arguments '
             'were not specified'},
        description="Trains a Blackbox Anomaly Detection Model for an entity with the specified entity_id. "
                    "The entity has to be already created. The Blackbox Model will be trained with the "
                    "uploaded file. The process of training will be asynchronous and an URL will be "
                    "returned in order to see the training progress.")
    def post(self, entity_id):
        """Trains a Blackbox model"""
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if not json_entities or entity_id not in json_entities:
            return {'error': 'The entity does not exist!'}, 400

        if not request.files:
            return {'error': 'No file was provided to train the model for the entity {}'.format(entity_id)}, 400

        if request.form and request.form.get('input_arguments'):
            input_arguments = request.form.get('input_arguments').split(',')
        else:
            return {
                       'error': 'Input arguments were not specified for the model'
                   }, 400

        if request.form and request.form.get('name'):
            model_name = request.form.get('name')
        else:
            date = datetime.now()
            model_name = 'model_{}_{}-{}-{}-{}:{}'.format(entity_id, date.year, date.month, date.day, date.hour,
                                                          date.minute)

        if request.form and request.form.get('models'):
            models = request.form.get('models').split(',')
        else:
            models = BlackBoxAnomalyDetection.AVAILABLE_MODELS

        # save the file
        file = request.files['file']
        filename, ext = os.path.splitext(file.filename)
        if ext != '.csv':
            return {'error': 'The file is not a .csv file'}, 400

        file.save(os.path.join(settings.MODELS_ROUTE, entity_id, 'train_data', secure_filename(file.filename)))

        # train the model
        path_train_file = settings.MODELS_ROUTE + '/' + entity_id + '/train_data/' + file.filename
        task = celery_app.send_task('tasks.train', args=[entity_id, path_train_file, model_name])

        return {
                   'message': 'The file was {} uploaded. Training model for entity {}'.format(file.filename, entity_id),
                   'task_status_url': build_url(request.url_root, settings.API_ANOMALY_ENDPOINT, 'task', task.id)
               }, 202


@anomaly_ns.route('/predict')
class Predict(Resource):
    @cors.crossdomain(origin='*')
    @anomaly_ns.doc(
        responses={202: 'Success', 400: 'No payload, the entity or the JSON file does not exist or an attr is missing'},
        description="This endpoint will receive the data from Orion Context Broker (FIWARE), i.e this endpoint has to "
                    "be specified in the HTTP URL field of a OCB subscription (OCB will make a POST to this endpoint)."
                    " With the data received from an entity, a prediction will be made using the default pre-trained "
                    "model for the entity.")
    def post(self):
        """Endpoint to receive data and predict if it's an anomaly."""
        if not request.json:
            return {'error': 'No payload in request'}, 400

        data = request.json['data'][0]
        entity_id = data['id']
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if not json_entities or entity_id not in json_entities:
            return {'error': 'The entity does not exists'}, 400

        entity = json_entities[entity_id]
        if (entity['default'] is None and len(entity['models']) == 0) or (len(entity['models']) == 0):
            return {'error': 'The entity has not trained models'}, 400

        default = entity['default']
        if default is None:  # if the default model is not set, take the first model from the dict of models
            entity_models = entity['models']
            first_model = list(entity_models.keys())[0]
            model_path = entity_models[first_model]['model_path']
        else:
            model_path = entity['models'][default]['model_path']

        predict_data = []
        for attr in entity['attrs']:
            try:
                date = data[attr]['metadata']['dateModified']['value']
                predict_data.append(data[attr]['value'])
            except KeyError:
                return {
                           'error': 'The attr {} was not in the sent attrs'.format(attr)
                       }, 400

        # parse date
        date = parser.parse(date).strftime("%Y-%m-%d %H:%M:%S")
        model_path = model['model_path']
        task = celery_app.send_task('tasks.predict', args=[entity_id, date, model_path, predict_data])

        return {
                   'message': 'The prediction for {} is being made...',
                   'task_status_url': build_url(request.url_root, settings.API_ANOMALY_ENDPOINT, 'task', task.id)
               }, 202


@anomaly_ns.route('/task/<string:task_id>')
@anomaly_ns.param('task_id', 'Celery task id')
class TaskStatus(Resource):
    @cors.crossdomain(origin='*')
    @anomaly_ns.doc(responses={200: 'Success'},
                    description="Returns the progress of the task specifying the state, the current progress, the "
                                "total progress that has to be reached and the status. Also, if the task has ended "
                                "a result will be returned too.")
    def get(self, task_id):
        """Gets the status of a task"""
        task = celery_app.AsyncResult(task_id)
        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'current': 0,
                'total': 100,
                'status': 'Pending...'
            }
        elif task.state != 'FAILURE':
            response = {
                'state': task.state,
                'current': task.info.get('current', 0),
                'total': task.info.get('total', 100),
                'status': task.info.get('status', '')
            }
            if 'result' in task.info:
                response['result'] = task.info['result']
        else:
            response = {
                'state': task.state,
                'current': 100,
                'total': 100,
                'status': str(task.info)
            }

        return response, 200


@anomaly_ns.errorhandler
def handle_root_exception(error):
    """Namespace error handler"""
    return {'error': str(error)}, getattr(error, 'code', 500)


def run_api():
    """
    Runs the API with the configuration inside the config file.
    """
    app.run(host=settings.APP_HOST, port=settings.APP_PORT, debug=settings.APP_DEBUG)


if __name__ == '__main__':
    run_api()
