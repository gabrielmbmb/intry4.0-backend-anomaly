import os
import settings
from flask import Flask, request
from flask_restplus import Api, Resource, cors
from werkzeug.utils import secure_filename
from blackbox.api.api_utils import read_json, write_json, add_entity_json
from blackbox.api.async_tasks import train_blackbox

# Create Flask App
app = Flask(settings.APP_NAME)
api = Api(app, version=settings.APP_VERSION, title=settings.APP_NAME, description=settings.APP_DESC)

# API Namespaces
anomaly_ns = api.namespace('api/v1/anomaly', description='Anomaly Detection Operations')

# API Routes
@anomaly_ns.route('/')
class ModelsList(Resource):
    @cors.crossdomain(origin='*')
    def get(self):
        """Returns a list of entities and its prediction models."""
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        return json_entities, 200


@anomaly_ns.route('/<string:entity_id>')
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
        created, msg = add_entity_json(settings.MODELS_ROUTE_JSON, entity_id,
                                       os.path.join(settings.MODELS_ROUTE, entity_id))
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


@anomaly_ns.route('/train/<string:entity_id>')
@anomaly_ns.param('entity_id', 'Orion Context Broker (FIWARE) entity ID')
class Train(Resource):
    @cors.crossdomain(origin='*')
    def post(self, entity_id):
        """Trains a Blackbox model for an entity with the data uploaded."""
        json_entities = read_json(settings.MODELS_ROUTE_JSON)
        if not json_entities or entity_id not in json_entities:
            created, msg = add_entity_json(settings.MODELS_ROUTE_JSON, entity_id,
                                           os.path.join(settings.MODELS_ROUTE, entity_id))

            if not created:
                return {'error': msg}, 400

        if not request.files:
            return {'error': 'No file was provided to train the model for the entity {}'.format(entity_id)}, 400

        # save the file
        file = request.files['file']
        file.save(os.path.join(settings.MODELS_ROUTE, entity_id, 'train_data', secure_filename(file.filename)))

        # train the model
        train_blackbox()

        return {
                   'message': 'The file was {} uploaded. Training model for entity {}.' \
                              + 'You can see the progress in the url'.format(file.filename, entity_id),
                   'task_url': 'url'
               }, 202


@anomaly_ns.route('/predict')
class Predict(Resource):
    @cors.crossdomain(origin='*')
    def post(self):
        pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678)
