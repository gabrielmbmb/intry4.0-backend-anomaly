import os
import settings
from dateutil import parser
from datetime import datetime
from flask import Flask, request
from flask_restplus import Api, Resource, cors, fields
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from blackbox.api.api_utils import read_json, add_entity_json, build_url, update_entity_json, \
    delete_entity_json
from blackbox.api.async_tasks import train_blackbox, predict_blackbox

# Todo: add logging to the Flask API

# Create Flask App
app = Flask(settings.APP_NAME)
api = Api(app, version='1.0', title=settings.APP_NAME, description=settings.APP_DESC)

# API Namespaces
anomaly_ns = api.namespace(settings.API_ANOMALY_ENDPOINT, description='Anomaly Detection Operations')

# API parsers
file_parser = anomaly_ns.parser()
file_parser.add_argument('file', type=FileStorage, required=True, location='files', help='CSV training file')

# API Models
entity_attrs_model = anomaly_ns.model('attrs', {
    'attrs': fields.List(fields.String(), description='New entity attributes', required=True)
})

update_entity_model = anomaly_ns.model('model', {
    'model_path': fields.String(description='Path of the model', required=False),
    'train_data_path': fields.String(description='Path of the training data', required=False),
})

update_entity_models = anomaly_ns.model('models', {
    'model': fields.Nested(update_entity_model)
})

update_entity = anomaly_ns.model('entity', {
    'new_entity_id': fields.String(description='New entity id', required=False),
    'default': fields.String(description='New default model', required=False),
    'attrs': fields.List(fields.String(), description='New entity attributes', required=False),
    'models': fields.Nested(update_entity_models)
})

# API Routes
@anomaly_ns.route('/')
@anomaly_ns.route('/entities')
class ModelsList(Resource):
    @cors.crossdomain(origin='*')
    def get(self):
        """Returns a list of entities and its prediction models."""
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if not json_entities:
            json_entities = {}

        return json_entities, 200


@anomaly_ns.route('/entity/<string:entity_id>')
@anomaly_ns.param('entity_id', 'Orion Context Broker (FIWARE) entity ID')
class Entity(Resource):
    @cors.crossdomain(origin='*')
    @anomaly_ns.response(200, 'Success')
    @anomaly_ns.response(400, 'Entity or JSON file does not exist')
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
    @anomaly_ns.doc(body=entity_attrs_model)
    @anomaly_ns.response(200, 'Success')
    @anomaly_ns.response(400, 'No payload, unable to create the entity or validation error')
    def post(self, entity_id):
        """Creates an entity"""
        if not request.json:
            return {'error': 'No payload was send'}, 400

        try:
            attrs = request.json['attrs']
        except KeyError:
            return {'error': 'No payload with attrs was send'}, 400
        except TypeError:
            return {'error': 'No payload with attrs was send'}, 400

        if not isinstance(attrs, list):
            return {'error': 'attrs has to be a list with strings inside'}, 400

        created, msg = add_entity_json(settings.MODELS_ROUTE_JSON, entity_id,
                                       os.path.join(settings.MODELS_ROUTE, entity_id), attrs)
        if not created:
            return {'error': msg}, 400

        return {'message': msg}, 200

    @cors.crossdomain(origin='*')
    @anomaly_ns.doc(body=update_entity)
    @anomaly_ns.response(200, 'Success')
    @anomaly_ns.response(400, 'No payload, unable to write or validation error')
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
    @anomaly_ns.response(200, 'Success')
    @anomaly_ns.response(400, 'Entity or JSON file does not exist')
    def delete(self, entity_id):
        """Deletes an entity."""
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
    @anomaly_ns.expect(file_parser)
    @anomaly_ns.response(202, 'Success')
    @anomaly_ns.response(400, 'Entity or JSON file does not exist or no training file provided')
    def post(self, entity_id):
        """Trains a Blackbox model for an entity with the data uploaded."""
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if not json_entities or entity_id not in json_entities:
            return {'error': 'The entity does not exist!'}, 400

        if not request.files:
            return {'error': 'No file was provided to train the model for the entity {}'.format(entity_id)}, 400

        if request.args and 'name' in request.args:
            model_name = request.args['name']
        else:
            date = datetime.now()
            model_name = 'model_{}_{}-{}-{}-{}:{}'.format(entity_id, date.year, date.month, date.day, date.hour,
                                                          date.minute)

        # save the file
        file = request.files['file']
        filename, ext = os.path.splitext(file.filename)
        if ext != '.csv':
            return {'error': 'The file is not a .csv file'}, 400

        file.save(os.path.join(settings.MODELS_ROUTE, entity_id, 'train_data', secure_filename(file.filename)))

        # train the model
        path_train_file = settings.MODELS_ROUTE + '/' + entity_id + '/train_data/' + file.filename
        task = train_blackbox.apply_async(args=[entity_id, path_train_file, model_name])

        return {
                   'message': 'The file was {} uploaded. Training model for entity {}'.format(file.filename, entity_id),
                   'task_status_url': build_url(request.url_root, settings.API_ANOMALY_ENDPOINT, 'task', task.id)
               }, 202


@anomaly_ns.route('/predict')
class Predict(Resource):
    @cors.crossdomain(origin='*')
    @anomaly_ns.response(202, 'Success')
    @anomaly_ns.response(400, 'No payload, the entity or the JSON file does not exist or an attr is missing')
    def post(self):
        """Endpoint to receive data from Orion Context Broker (FIWARE) and predict if it's an anomaly."""
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
        task = predict_blackbox.apply_async(args=[entity_id, date, model_path, predict_data])

        return {
                    'message': 'The prediction for {} is being made...',
                    'task_status_url': build_url(request.url_root, settings.API_ANOMALY_ENDPOINT, 'task', task.id)
               }, 202


@anomaly_ns.route('/task/<string:task_id>')
@anomaly_ns.param('task_id', 'Celery task id')
class TaskStatus(Resource):
    @cors.crossdomain(origin='*')
    @anomaly_ns.response(200, 'Success')
    def get(self, task_id):
        """Gets the status of a task"""
        task = train_blackbox.AsyncResult(task_id)
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
    return {'error': error}, 404
