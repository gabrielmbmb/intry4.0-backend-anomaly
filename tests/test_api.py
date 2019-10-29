import os
import shutil
from unittest import TestCase
from datetime import datetime
from blackbox.api.api import app
from blackbox.api.utils import read_json


class TestFlaskApi(TestCase):
    """Test Flask API"""

    API_ANOMALY_ENDPOINT = '/api/v1'
    TEST_MODELS_JSON_PATH = os.path.expanduser('~/blackbox/models/models.json')

    def setUp(self):
        self.app = app.test_client()

    def tearDown(self):
        dirpath = os.getcwd()
        os.chdir(os.path.expanduser('~'))
        shutil.rmtree('./blackbox')
        os.chdir(dirpath)

    def test_create_entity(self):
        """Test if an entity is correctly created"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        response = self.app.post(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id, json={"attrs": ["Bearing1"]})

        self.assertEqual(response.status_code, 200)

        json_entities = read_json(self.TEST_MODELS_JSON_PATH)
        entity = json_entities[entity_id]
        self.assertDictEqual(entity, {
            "attrs": ["Bearing1"],
            "default": None,
            "models": {}
        })

    def test_get_entity(self):
        """Test getting an entity from the JSON file"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        self.app.post(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id, json={"attrs": ["Bearing1"]})

        entity = {
            entity_id: {
                "attrs": [
                    "Bearing1",
                ],
                "default": None,
                "models": {}
            }
        }

        response = self.app.get(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id)
        self.assertEqual(response.status_code, 200)
        self.assertDictEqual(response.json, entity)

    def test_create_entity_already_existing(self):
        """Test creating an entity which already exists"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        self.app.post(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id, json={"attrs": ["Bearing1"]})
        response = self.app.post(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id, json={"attrs": ["Bearing1"]})
        self.assertEqual(response.status_code, 400)

    def test_update_entity(self):
        """Test if an entity is correctly updated"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        self.app.post(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id, json={"attrs": ["Bearing1"]})

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

        response = self.app.put(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id, json=data_update)
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

    def test_update_entity_id_already_exist(self):
        """Test updating an entity id with one that already exists"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        self.app.post(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id, json={"attrs": ["Bearing1"]})

        entity_id_2 = 'urn:ngsi-ld:Machine:002'
        self.app.post(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id_2, json={"attrs": ["Bearing1"]})

        data_update = {"new_entity_id": entity_id}
        response = self.app.put(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id, json=data_update)
        self.assertEqual(response.status_code, 400)

    def test_delete_entity(self):
        """Test if an entity is correctly deleted"""
        entity_id = 'urn:ngsi-ld:Machine:001'
        self.app.post(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id, json={"attrs": ["Bearing1"]})
        response = self.app.delete(self.API_ANOMALY_ENDPOINT + '/entities/' + entity_id)
        self.assertEqual(response.status_code, 200)
        date = datetime.now()
        date_string = '{}-{}-{}-{}:{}'.format(date.year, date.month, date.day, date.hour, date.minute)
        self.assertTrue(os.path.exists(os.path.expanduser('~/blackbox/models/trash/') + entity_id + '_' + date_string))

    # def test_train_entity(self):
    #     """Test training a Blackbox model for an entity"""
    #     entity_id = 'urn:ngsi-ld:Machine:001'
    #     self.app.post(self.API_ANOMALY_ENDPOINT + '/entity/' + entity_id,
    #                   json={"attrs": ["Bearing1", "Bearing2", "Bearing3", "Bearing4"]})
    #
    #     response = self.app.post(self.API_ANOMALY_ENDPOINT + '/train/' + entity_id,
    #                              content_type='multipart/form-data',
    #                              data={'file': open('./tests/train_data.csv', 'rb')})
    #
    #     self.assertEqual(response.status_code, 202)
