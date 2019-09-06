import os
from unittest import TestCase
from settings import API_ANOMALY_ENDPOINT
from blackbox.api.api import app
from blackbox.api.api_utils import read_json


class TestFlaskApi(TestCase):
    """Test Flask API"""

    TEST_MODELS_PATH = './models'
    TEST_MODELS_JSON_PATH = './models/models.json'

    def setUp(self) -> None:
        self.app = app.test_client()

    def test_a_create_entity(self) :
        """Test if an entity is correctly created"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        response = self.app.post(API_ANOMALY_ENDPOINT + '/entity/' + entity_id, json={
            "attrs": [
                "Bearing1"
            ]
        })

        self.assertEqual(response.status_code, 200)

        json_entities = read_json(self.TEST_MODELS_JSON_PATH)
        entity = json_entities[entity_id]
        self.assertDictEqual(entity, {
            "attrs": ["Bearing1"],
            "default": None,
            "models": {}
        })

    def test_b_get_entity(self):
        """Test getting an entity from the JSON file"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        entity = {
            entity_id: {
                "attrs": [
                    "Bearing1",
                ],
                "default": None,
                "models": {}
            }
        }

        response = self.app.get(API_ANOMALY_ENDPOINT + '/entity/' + entity_id)

        self.assertEqual(response.status_code, 200)

        self.assertDictEqual(response.json, entity)

    def test_c_create_entity_already_existing(self):
        """Test creating an entity which already exists"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        response = self.app.post('/api/v1/anomaly/entity/' + entity_id, json={
            "attrs": [
                "Bearing1"
            ]
        })

        self.assertEqual(response.status_code, 400)

    def test_d_update_entity(self):
        """Test if an entity is correctly updated"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        new_entity_id = 'urn:ngsi-ld:Machine:999'
        data_update = {
            "new_entity_id": new_entity_id,
            "default": "./models/urn:ngsi-ld:Machine:001/test_model.pkl",
            "attrs": [
                "Pressure1",
                "Pressure2"
            ],
            "models": {
                "model": {
                    "model_path": "./models/urn:ngsi-ld:Machine:001/test_model.pkl",
                    "train_data_path": "./models/urn:ngsi-ld:Machine:001/train_data/train_data.csv"
                }
            }
        }

        response = self.app.put('/api/v1/anomaly/entity/' + entity_id, json=data_update)

        self.assertEqual(response.status_code, 200)

        json_entities = read_json(self.TEST_MODELS_JSON_PATH)
        entity = json_entities[new_entity_id]

        self.assertDictEqual(entity, {
            "attrs": [
                "Pressure1",
                "Pressure2"
            ],
            "default": None,
            "models": {
                "model": {
                    "model_path": "./models/urn:ngsi-ld:Machine:001/test_model.pkl",
                    "train_data_path": "./models/urn:ngsi-ld:Machine:001/train_data/train_data.csv"
                }
            }
        })

    def test_e_delete_entity(self):
        """Test if an entity is correctly deleted"""
        entity_id = 'urn:ngsi-ld:Machine:999'
        response = self.app.delete(API_ANOMALY_ENDPOINT + '/entity/' + entity_id)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(os.path.exists('./models/trash/' + entity_id))
