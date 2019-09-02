import os
import json
from typing import Tuple

ENTITY_JSON_STRUCTURE = {"default": None, "models": {}}


def read_json(path):
    """
    Reads a JSON file.

    Args:
        path (str): path of the JSON file.

    Returns:
        dict with the content of the JSON file or None if the JSON couldn't be read.
    """
    try:
        json_ = json.load(open(path, 'r'))
    except FileNotFoundError as e:
        print('The file does not exists:', e)
        json_ = None

    return json_


def write_json(path, data) -> None:
    """
    Write a JSON file in the specified path.

    Args:
        path (str): path of the JSON file to be created.
        data (dict): data that is going to be written in the JSON file.
    """
    with open(path, 'w') as f:
        json.dump(data, f)


def add_entity_json(path_json, entity_id, path_entity_dir) -> Tuple[bool, str]:
    """
    Add an entity to the JSON file. If the JSON file is not created, then it will be created and add the entity will be
    added inside of it. If the entity already exist, the entity will not be created.

    Args:
        path (str): JSON models file path.
        entity_id (str): entity ID (Orion Context Broker)
        path_entity_dir (str): path of the entity directory.

    Returns:
        bool: indicates if the entity was created.
    """
    try:
        os.makedirs(os.path.join(path_entity_dir, 'train_data'))
    except FileExistsError as e:
        print('The directory {} already exists. Aborting...'.format(path_entity_dir))
        return False, 'The directory {} already exists.'.format(path_entity_dir)
    except OSError as e:
        print('Error creating directory {} for entity {}. Aborting...'.format(path_entity_dir, entity_id))
        return False, 'The directory {} could not be created'.format(path_entity_dir)

    json_entities = read_json(path_json)
    if not json_entities:
        entities = {entity_id: ENTITY_JSON_STRUCTURE}
        write_json(path_json, entities)
    else:
        if entity_id in json_entities:
            return False, 'The entity {} already exists'.format(entity_id)

        json_entities[entity_id] = ENTITY_JSON_STRUCTURE
        write_json(path_json, json_entities)

    return True, 'The entity {} was created'.format(entity_id)


def build_url(url_root, base_endpoint, *args):
    """
    Builds the complete URL for an API endpoint.

    Args:
        url_root (str): root URL. i.e: 'http://localhost:5678'
        base_endpoint (str): endpoint base. i.e: 'api/v1/anomaly'
        *args: arbitrary number of strings that will be added to the end of the URL. i.e: 'api', 'anomaly' ->
            'api/anomaly'

    Returns:
        str: the build URL.
    """
    complete_url = ''
    if (url_root[-1] == '/' and base_endpoint[0] != '/') or (url_root[-1] != '/' and base_endpoint[0] == '/'):
        complete_url = url_root + base_endpoint
    elif url_root[-1] == '/' and base_endpoint[0] == '/':
        complete_url = url_root[:-1] + base_endpoint
    elif url_root[-1] != '/' and base_endpoint[0] != '/':
        complete_url = url_root + '/' + base_endpoint

    if complete_url[-1] == '/':
        complete_url = complete_url[:-1]

    for point in args:
        complete_url = complete_url + '/' + point

    return complete_url
