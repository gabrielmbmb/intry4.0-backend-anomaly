import os
import settings
from datetime import datetime
from flask import Flask, request
from flask_restplus import Api, Resource, cors
from werkzeug.utils import secure_filename
from blackbox.api.api_utils import read_json, write_json, add_entity_json, build_url, update_entity_json
from blackbox.api.async_tasks import train_blackbox, predict_blackbox

# Create Flask App
app = Flask(settings.APP_NAME)
api = Api(app, version=settings.APP_VERSION, title=settings.APP_NAME, description=settings.APP_DESC)

# API Namespaces
anomaly_ns = api.namespace(settings.API_ANOMALY_ENDPOINT, description='Anomaly Detection Operations')

# API Routes
@anomaly_ns.route('/')
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
class Machine(Resource):

    @cors.crossdomain(origin='*')
    def get(self, entity_id):
        """Return an entity and its prediction models."""
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if not json_entities:
            return {'error': 'The JSON file does not exist'}, 400

        if entity_id not in json_entities:
            return {'error': 'The entity {} does not exist'.format(entity_id)}, 400

        entity_data = json_entities[entity_id]
        return {entity_id: entity_data}, 200

    @cors.crossdomain(origin='*')
    def post(self, entity_id):
        """Creates an entity in the JSON file."""
        if not request.json['attrs']:
            return {'error': 'No payload with entity attributes was send'}, 400

        attrs = request.json['attrs']

        created, msg = add_entity_json(settings.MODELS_ROUTE_JSON, entity_id,
                                       os.path.join(settings.MODELS_ROUTE, entity_id), attrs)
        if not created:
            return {'error': msg}, 400

        return {'message': msg}, 200

    @cors.crossdomain(origin='*')
    def delete(self, entity_id):
        """Deletes an entity."""
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if not json_entities:
            return {'error': 'The JSON file storing entities does not exist'}, 400

        if entity_id not in json_entities:
            return {'error': 'The entity {} does not exist'.format(entity_id)}, 400

        json_entities.pop(entity_id, False)

        write_json(settings.MODELS_ROUTE_JSON, json_entities)
        return {'message': 'The entity {} has been removed'.format(entity_id)}, 200


@anomaly_ns.route('/entity/update/<string:entity_id>')
@anomaly_ns.param('entity_id', 'Orion Context Broker (FIWARE) entity ID')
class UpdateEntity(Resource):
    @cors.crossdomain(origin='*')
    def post(self, entity_id):
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

        updated, messages = update_entity_json(entity_id, settings.MODELS_ROUTE_JSON, new_entity_id, default, attrs,
                                               models)

        if not updated:
            return {
                        'error': 'The entity was not updated',
                        'messages': messages
                   }, 400

        return {
                   'messages': messages
               }, 200


@anomaly_ns.route('/train/<string:entity_id>')
@anomaly_ns.param('entity_id', 'Orion Context Broker (FIWARE) entity ID')
class Train(Resource):
    @cors.crossdomain(origin='*')
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
    def post(self):
        """Endpoint to receive data from Orion Context Broker (FIWARE) and predict if it's an anomaly."""
        if not request.json:
            return {'error': 'No payload in request'}, 400

        data = request.json['data'][0]
        entity_id = data['id']
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if entity_id not in json_entities:
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
                predict_data.append(data[attr]['value'])
            except KeyError:
                return {
                            'error': 'The attr {} was not in the sent attrs'.format(attr)
                       }, 400

        task = predict_blackbox.apply_async(args=[entity_id, model_path, predict_data])

        return {
                    'message': 'The prediction for {} is being made...',
                    'task_status_url': build_url(request.url_root, settings.API_ANOMALY_ENDPOINT, 'task', task.id)
               }, 202


@anomaly_ns.route('/task/<string:task_id>')
class TaskStatus(Resource):
    @cors.crossdomain(origin='*')
    def get(self, task_id):
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678)
