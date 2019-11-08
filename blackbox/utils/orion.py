import requests
from typing import Union
from blackbox.settings import FIWARE_SERVICE, FIWARE_SERVICEPATH, ORION_CONTEXT_BROKER


def check_orion_connection() -> bool:
    """
    Checks if the Orion Context Broker is up.

    Returns:
        bool: indicates if the OCB is up or not.
    """
    url = ORION_CONTEXT_BROKER + '/version'
    try:
        requests.get(url)
    except requests.exceptions.RequestException:
        return False

    return True


def update_entity_attrs(entity_id, attrs) -> Union[None, requests.Response]:
    """
    Updates entity attributes stored in Orion Context Broker.

    Args:
        entity_id (str): entity ID in Orion Context Broker
        attrs (dict): dictionary containing the attributes that will be created or updated in the entity.

    Returns:
    """
    if not isinstance(attrs, dict):
        print('Attributes passed are not in a dictionary')
        return None

    url = '{}/v2/entities/{}/attrs'.format(ORION_CONTEXT_BROKER, entity_id)

    try:
        response = requests.post(url, json=attrs, headers={
            'fiware-servicepath': FIWARE_SERVICEPATH,
            'fiware-service': FIWARE_SERVICE,
            'Content-Type': 'application/json'
        })
    except requests.exceptions.RequestException:
        response = None

    return response
