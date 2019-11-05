from unittest import TestCase
from blackbox.utils import api


class TestApiUtils(TestCase):
    """Test API """

    TEST_URL = 'http://localhost:5678/api/v1/task/random-task-1234'
    REGEXES = ['urn:ngsi-ld:Table:[0-9]{3}$', 'urn:ngsi-ld:Machine:[0-9]{5}$']

    def test_build_url(self):
        """Tests if the URL is correctly constructed"""
        self.assertEqual(self.TEST_URL, api.build_url('http://localhost:5678/', 'api/v1/', 'task', 'random-task-1234'))
        self.assertEqual(self.TEST_URL, api.build_url('http://localhost:5678', '/api/v1/', 'task', 'random-task-1234'))
        self.assertEqual(self.TEST_URL, api.build_url('http://localhost:5678/', '/api/v1/', 'task', 'random-task-1234'))
        self.assertEqual(self.TEST_URL, api.build_url('http://localhost:5678', 'api/v1/', 'task', 'random-task-1234'))

    def test_match_regex(self):
        """Tests if regular expressions are matched"""
        self.assertEqual(self.REGEXES[0], api.match_regex(self.REGEXES, 'urn:ngsi-ld:Table:001'))
        self.assertEqual(self.REGEXES[1], api.match_regex(self.REGEXES, 'urn:ngsi-ld:Machine:00001'))
        self.assertEqual(None, api.match_regex(self.REGEXES, 'urn:ngsi-ld:Board:00001'))

    def test_parse_float(self):
        """Tests if a string is correctly parsed to float"""
        self.assertListEqual([0.00001, 0.2, 11.4, 11], api.parse_float(['0.00001', '0.2', '11.4', '11']))
        self.assertEqual(0.0002, api.parse_float('0.0002'))
        self.assertEqual([0.00001, 0.2, 11.4, 11], api.parse_float(['0.00001', '0.2', '11.4', '11', 't']))
        self.assertEqual([], api.parse_float(['0.00001t', '0.2e', '11.45h4', '1c1', 'et']))
        self.assertEqual(None, api.parse_float('t'))
