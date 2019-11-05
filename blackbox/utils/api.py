import os
import re
import json
from datetime import datetime
from typing import Tuple, List, Dict, Union

ENTITY_JSON_STRUCTURE = {"attrs": None, "default": None, "models": {}}


def read_json(path) -> Dict:
    """
    Reads a JSON file.

    Args:
        path (str): path of the JSON file.

    Returns:
        dict: dict with the content of the JSON file or None if the JSON couldn't be read.
    """
    try:
        with open(path, 'r') as f:
            json_ = json.load(f)
    except FileNotFoundError as e:
        json_ = None

    return json_


def write_json(path, data, sort=False) -> None:
    """
    Write a JSON file in the specified path.

    Args:
        path (str): path of the JSON file to be created.
        data (dict): data that is going to be written in the JSON file.
        sort (bool): indicates if the JSON has to be sorted by key before writing it.
    """
    if sort:
        keys_sorted = sorted(data)

        new_json = {}
        for key in keys_sorted:
            new_json[key] = data[key]

        data = new_json

    with open(path, 'w') as f:
        json.dump(data, f)


def add_entity_json(path_json, entity_id, path_entity_dir, attrs) -> Tuple[bool, str]:
    """
    Add an entity to the JSON file. If the JSON file is not created, then it will be created and add the entity will be
    added inside of it. If the entity already exist, the entity will not be created.

    Args:
        path_json (str): JSON models file path.
        entity_id (str): entity ID (Orion Context Broker)
        path_entity_dir (str): path of the entity directory.
        attrs (list of str): attributes of the entity (same name as in Orion Context Broker).

    Returns:
        tuple: bool which indicates if the entity was created and str which is a descriptive message.
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
        json_entities = {entity_id: ENTITY_JSON_STRUCTURE}
    else:
        if entity_id in json_entities:
            return False, 'The entity {} already exists'.format(entity_id)

        json_entities[entity_id] = ENTITY_JSON_STRUCTURE

    json_entities[entity_id]['attrs'] = attrs
    write_json(path_json, json_entities, sort=True)

    return True, 'The entity {} was created'.format(entity_id)


def add_model_entity_json(path_json, entity_id, model_name, model_path, train_data_path, models,
                          input_arguments) -> bool:
    """
    Adds a model of an entity to the JSON file.

    Args:
        path_json (str): JSON models file path.
        entity_id (str): entity ID (Orion Context Broker).
        model_name (str): model name.
        model_path (str): path of the pickle file storing the Anomaly Detection model.
        train_data_path (str): path of the file used to train the model.
        models (list of str): contains the names of the Anomaly Detection models that are inside the Blackbox.
        input_arguments (list of str): name of the inputs variables of the Blackbox model.


    Returns:
        bool: indicating whether the model was added or not.

    Raises:
        FileNotFoundError: if the JSON file storing the entities and its models does not exist.
    """
    json_entities = read_json(path_json)
    if not json_entities:
        raise FileNotFoundError('File {} does not exists!'.format(path_json))

    try:
        entity_dict = json_entities[entity_id]
    except KeyError as e:
        print('Entity does not exist: ', e)
        return False

    if entity_dict['default'] is None:
        entity_dict['default'] = model_name

    entity_dict['models'][model_name] = {
        'input_arguments': input_arguments,
        'models': models,
        'model_path': model_path,
        'train_data_path': train_data_path
    }

    json_entities[entity_id] = entity_dict
    write_json(path_json, json_entities, sort=True)
    return True


def update_entity_json(entity_id, path_json, path_models, new_entity_id=None,
                       default=None, attrs=None, models=None) -> Tuple[bool, List[str]]:
    """
    Updates an entity stored in the JSON file.

    Args:
        entity_id (str): entity ID (Orion Context Broker).
        path_json (str): path of the JSON file storing the entities info.
        path_models (str): path where the models are stored.
        new_entity_id (str): new entity ID (Orion Context Broker) that will replace entity_id. Defaults to None.
        default (str): new default model. Defaults to None.
        attrs (list of str): new attributes of the entity (same name as in Orion Context Broker).
        models (dict): new models. Defaults to None.

    Returns:
        tuple: bool which indicates if the entity was updated and list of str containing descriptive messages.

    Raises:
        FileNotFoundError: if the JSON file storing the entities and its models does not exist.
    """
    json_entities = read_json(path_json)
    if not json_entities:
        raise FileNotFoundError('File {} does not exists!'.format(path_json))

    messages = []
    updated = False

    if entity_id not in json_entities:
        messages.append('The entity does not exist.')
        return updated, messages

    entity = json_entities[entity_id]

    if models:
        if isinstance(models, dict):
            entity['models'] = models
            messages.append('The parameter models has been updated.')
            updated = True
        else:
            messages.append('The parameter models has to be a dict.')

    if default:
        if isinstance(default, str):
            if default in entity['models']:
                entity['default'] = default
                messages.append('The parameter default has been updated.')
                updated = True
            else:
                messages.append(
                    'The model {} does not exists for entity {} and cannot be set as default model'.format(default,
                                                                                                           entity_id))
        else:
            messages.append('The parameter default has to be an str.')

    if attrs:
        if isinstance(attrs, list):
            entity['attrs'] = attrs
            messages.append('The parameter attrs has been updated.')
            updated = True
        else:
            messages.append('The parameter attrs has to be a list of str.')

    json_entities[entity_id] = entity

    if new_entity_id:
        if new_entity_id in json_entities:
            messages.append('An entity already exist with id {}'.format(new_entity_id))
        else:
            if isinstance(new_entity_id, str):
                json_entities[new_entity_id] = entity
                json_entities.pop(entity_id, None)
                os.rename(os.path.join(path_models, entity_id), os.path.join(path_models, new_entity_id))
                messages.append('The parameter entity_id has been updated.')
                updated = True
            else:
                messages.append('The parameter new_entity_id has to be an str')

    if updated:
        write_json(path_json, json_entities, sort=True)

    return updated, messages


def delete_entity_json(entity_id, path_json, path_models, path_trash) -> Tuple[bool, str]:
    """
    Deletes an entity from the JSON storing the entities and its models. The directory of the entity will be moved
    to the deleted entities folder.

    Args:
        entity_id (str): entity ID (Orion Context Broker).
        path_json (str): path of the JSON file storing the entities info.
        path_models (str): path where the models are stored.
        path_trash (str): path where the directory of the deleted entities will be stored.

    Returns:
        tuple: tuple containing one bool indicating if the entity was deleted and str that is a descriptive message.
    """
    if not os.path.exists(path_trash):
        os.mkdir(path_trash)

    json_entities = read_json(path_json)
    if not json_entities:
        raise FileNotFoundError('File {} does not exists!'.format(path_json))

    if entity_id not in json_entities:
        return False, 'The entity {} does not exists.'.format(entity_id)

    json_entities.pop(entity_id, False)
    write_json(path_json, json_entities)

    # move the dir of the entity to the trash folder
    date = datetime.now()
    date_string = '{}-{}-{}-{}:{}'.format(date.year, date.month, date.day, date.hour, date.minute)
    os.rename(path_models + '/' + entity_id, path_trash + '/' + entity_id + '_' + date_string)

    return True, 'The entity {} was deleted and its directory was moved to {}'.format(entity_id, path_trash)


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


def match_regex(regexes, string) -> Union[str, None]:
    """
    Checks if the string matches any regular expression contained in regexes.

    Args:
        regexes (list of str): contains regular expressions
        string (str): string to check if matches with any regular expression.

    Returns:
        str: the regular expression that has matched
    """
    for regex in regexes:
        if re.match(regex, string):
            return regex

    return None


def parse_float(to_parse) -> Union[float, list, None]:
    """
    Parses a string and returns a floating point number

    Args:
        to_parse(list or str): a single string or a list of strings.

    Returns:
        float or list: floating point number parsed from a string or a list of floatings point numbers parsed from a
            list of strings
    """
    if isinstance(to_parse, list):
        parsed_floats = []
        for string in to_parse:
            try:
                parsed_floats.append(float(string))
            except ValueError:
                print('String {} could not be parsed to float'.format(parsed_floats))

        return parsed_floats

    try:
        parsed_float = float(to_parse)
    except ValueError:
        parsed_float = None
        print('String {} could not be parsed to float'.format(parsed_float))

    return parsed_float
