import json
from flask import Flask, request
from flask_restplus import Api, Resource, cors

MODELS_ROUTE = './api/models'
MODELS_ROUTE_JSON = MODELS_ROUTE + '/models.json'

app = Flask(__name__)
api = Api(app=app,
          version='1.0',
          title='PLATINUM - Blackbox Anomaly Detection',
          description='A simple API to call the Blackbox Anomaly Detection model.')

anomaly_ns = api.namespace('api/v1/anomaly', description='Anomaly Detection Operations')


@anomaly_ns.route('/')
class ModelsList(Resource):
    @cors.crossdomain(origin='*')
    def get(self):
        """Returns a list of entities and its prediction models."""
        json_entities = read_json(MODELS_ROUTE_JSON)
        return json_entities, 200


@anomaly_ns.route('/<string:entity_id>')
@anomaly_ns.param('entity_id', 'Orion Context Broker (FIWARE) entity ID')
class Machine(Resource):

    @cors.crossdomain(origin='*')
    def get(self, entity_id):
        """Return an entity and its prediction models."""
        json_entities = read_json(MODELS_ROUTE_JSON)
        if entity_id not in json_entities:
            return {'error': 'The entity {} does not exist'.format(entity_id)}, 400

        entity_data = json_entities[entity_id]
        return {entity_id: entity_data}, 200

    @cors.crossdomain(origin='*')
    def post(self, id):
        """Creates an entity in the JSON file."""
        json_entities = read_json(MODELS_ROUTE_JSON)

        if id in json_entities:
            return {'error': 'The entity {} does already exist'.format(id)}, 400

        json_entities[id] = {
            "default": None,
            "models": []
        }

        write_json(MODELS_ROUTE_JSON, json_entities)
        return {'message': 'The entity {} has been created'.format(id)}, 200

    @cors.crossdomain(origin='*')
    def delete(self, id):
        """Deletes an entity."""
        json_entities = read_json(MODELS_ROUTE_JSON)

        if id not in json_entities:
            return {'error': 'The entity {} does not exist'.format(id)}, 400

        json_entities.pop(id, False)

        write_json(MODELS_ROUTE_JSON, json_entities)
        return {'message': 'The entity {} has been removed'.format(id)}, 200


@anomaly_ns.route('/train')
class Train(Resource):
    @cors.crossdomain(origin='*')
    def post(self):
        pass


@anomaly_ns.route('/predict')
class Predict(Resource):
    @cors.crossdomain(origin='*')
    def post(self):
        pass


def read_json(path):
    with open(path, 'r') as f:
        json_ = json.load(f)

    return json_


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5678)


