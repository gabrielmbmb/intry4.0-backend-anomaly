#!/usr/bin/env python3

import click
import requests
import json

from pygments import highlight
from pygments.lexers.data import JsonLexer
from pygments.formatters.terminal import TerminalFormatter

DEFAULT_FIWARE_HOST = 'http://localhost:1026'
DEFAULT_BLACKBOX_HOST = 'http://localhost:5678'
DEFAULT_FIWARE_SERVICE = 'sensores'
DEFAULT_TYPE = 'Machine'


@click.group()
@click.option('--on', type=click.Choice(['fiware', 'blackbox', 'both']), default='both',
              help='Indicates where the operation has to be done.')
@click.option('--blackbox_url', '-ah', default=DEFAULT_BLACKBOX_HOST, help='URL of Blackbox API')
@click.option('--fiware_url', '-fh', default=DEFAULT_FIWARE_HOST, help='URL of Orion Context Broker (FIWARE)')
@click.option('--fw_service', '-fs', default=DEFAULT_FIWARE_SERVICE, help='Orion Context Broker service path')
@click.option('--entity_type', '-et', default=DEFAULT_TYPE, help='Type of the entity')
@click.pass_context
def entity(ctx, on, blackbox_url, fiware_url, fw_service, entity_type):
    ctx.ensure_object(dict)
    ctx.obj['ON'] = on
    ctx.obj['BLACKBOX_HOST'] = blackbox_url
    ctx.obj['FIWARE_HOST'] = fiware_url
    ctx.obj['FIWARE_SERVICE'] = fw_service
    ctx.obj['TYPE'] = entity_type


############
# ENTITIES #
############

def parse_attrs(attrs) -> dict:
    """
    Parses a list of str with the format "Name,Type,Value" into a dictionary.

    Args:
        attrs (list of str): attributes

    Returns:
        dict: dictionary with the parsed attributes
    """
    attr_dict = {}
    for attr in attrs:
        attr_split = attr.split(',')
        if len(attr_split) != 3:
            click.echo('Badly constructed attribute: {}'.format(attr))
            continue

        attr_split[1] = attr_split[1].title()
        if attr_split[1] == 'Float':
            attr_split[2] = float(attr_split[2])
        elif attr_split[1] == 'Text':
            pass
        elif attr_split[1] == 'Integer':
            attr_split[2] = int(attr_split[2])
        else:
            click.echo('Invalid type for attribute {}: {}'.format(attr_split[0], attr_split[1]))

        attr_dict[attr_split[0]] = {'type': attr_split[1], 'value': attr_split[2]}

    return attr_dict


def beautify_json(json_data) -> dict:
    """
    Beautify a JSON.

    Args:
        json_data (dict): JSON.

    Returns:
        dict: beautified JSON.
    """
    formatted_json = json.dumps(json_data, indent=4)
    highlighted_json = highlight(formatted_json, JsonLexer(), TerminalFormatter())
    return highlighted_json


@entity.command('create_entity')
@click.argument('entity_id')
@click.argument('attrs', nargs=-1)
@click.pass_context
def create_entity(ctx, entity_id, attrs):
    """
    Creates an entity in Orion Context Broker (FIWARE) and in the Blackbox Anomaly Detection Model.

    Args:
        ctx (object): click object
        entity_id (str): Orion Context Broker (FIWARE) entity id
        attrs (list of str): list of strings containing the name, type and value of the attributes.
    """
    click.echo('Creating {} in {} & {} (service: {})'.format(entity_id, ctx.obj['BLACKBOX_HOST'],
                                                             ctx.obj['FIWARE_HOST'], ctx.obj['FIWARE_SERVICE']))

    attrs_dict = {}
    if attrs:
        attrs_dict = parse_attrs(attrs)

    # create entity in Orion Context Broker
    if ctx.obj['ON'] == 'fiware' or ctx.obj['ON'] == 'both':
        url = ctx.obj['FIWARE_HOST'] + '/v2/entities'
        data = {**{'id': entity_id, 'type': ctx.obj['TYPE']}, **attrs_dict}
        headers = {
            'Content-Type': 'application/json',
            'fiware-service': ctx.obj['FIWARE_SERVICE'],
            'fiware-servicepath': '/',
        }
        try:
            response = requests.post(url=url, headers=headers, json=data)
            click.echo('[FIWARE] STATUS_CODE: {}'.format(response.status_code))
        except requests.exceptions.ConnectionError:
            click.echo('Error connecting with Orion Context Broker')

    # create entity in Blackbox API
    if ctx.obj['ON'] == 'blackbox' or ctx.obj['ON'] == 'both':
        url = ctx.obj['BLACKBOX_HOST'] + '/api/v1/anomaly/entity/' + entity_id
        attrs = [key for key in attrs_dict.keys()]
        try:
            response = requests.post(url=url, json={'attrs': attrs})
            click.echo(
                '[BLACKBOX API] STATUS_CODE: {}, MSG: {}'.format(response.status_code, beautify_json(response.json())))
        except requests.exceptions.ConnectionError:
            click.echo('Error connecting with Blackbox API')


@entity.command('delete_entity')
@click.argument('entity_id')
@click.pass_context
def delete_entity(ctx, entity_id):
    """
    Deletes an entity in Orion Context Broker (FIWARE) and in the Blackbox Anomaly Detection Model.

    Args:
        ctx (object): click object
        entity_id (str): Orion Context Broker (FIWARE) entity id
    """
    click.echo('Deleting {} in {} & {} (service: {})'.format(entity_id, ctx.obj['BLACKBOX_HOST'],
                                                             ctx.obj['FIWARE_HOST'], ctx.obj['FIWARE_SERVICE']))

    # deletes an entity in Orion Context Broker
    if ctx.obj['ON'] == 'fiware' or ctx.obj['ON'] == 'both':
        url = ctx.obj['FIWARE_HOST'] + '/v2/entities/' + entity_id
        headers = {'fiware-service': ctx.obj['FIWARE_SERVICE'], 'fiware-servicepath': '/'}
        try:
            response = requests.delete(url=url, headers=headers)
            click.echo('[FIWARE] STATUS_CODE: {}'.format(response.status_code))
        except requests.exceptions.ConnectionError:
            click.echo('Error connecting with Orion Context Broker')

    # deletes an entity in Blackbox API
    if ctx.obj['ON'] == 'blackbox' or ctx.obj['ON'] == 'both':
        url = ctx.obj['BLACKBOX_HOST'] + '/api/v1/anomaly/entity/' + entity_id
        try:
            response = requests.delete(url=url)
            click.echo(
                '[BLACKBOX API] STATUS_CODE: {}, MSG: {}'.format(response.status_code, beautify_json(response.json())))
        except requests.exceptions.ConnectionError:
            click.echo('Error connecting with Blackbox API')


@entity.command('update_entity')
@click.argument('attrs', nargs=-1)
@click.pass_context
def update_entity(ctx, attrs):
    """
    Updates the passed attributes of an entity in Orion Context Broker (FIWARE) and Blackbox API.
    i.e: "Bearing1,Float,0.67"

    Args:
        ctx (object): click object
        attrs (list of str): list of strings containing the name, type and value of the attributes.
    """

    # parse attributes
    attr_dict = parse_attrs(attrs)

    # add attributes to Orion Context Broker
    if ctx.obj['ON'] == 'fiware' or ctx.obj['ON'] == 'both':
        url = ctx.obj['FIWARE_HOST'] + '/v2/entities/' + ctx.obj['ENTITY_ID'] + '/attrs'
        headers = {
            'Content-Type': 'application/json',
            'fiware-service': ctx.obj['FIWARE_SERVICE'],
            'fiware-servicepath': '/'
        }
        try:
            response = requests.post(url=url, headers=headers, json=attr_dict)
            click.echo('[FIWARE] STATUS_CODE: {}'.format(response.status_code))
        except requests.exceptions.ConnectionError:
            click.echo('Error connecting with Orion Context Broker')

    # add attributes to Blackbox API
    if ctx.obj['ON'] == 'blackbox' or ctx.obj['ON'] == 'both':
        url = ctx.obj['BLACKBOX_HOST'] + '/api/v1/anomaly/entity/' + ctx.obj['ENTITY_ID']
        attrs = [key for key in attr_dict.keys()]
        try:
            response = requests.put(url=url, json={'attrs': attrs})
            click.echo(
                '[BLACKBOX API] STATUS_CODE: {}, MSG: {}'.format(response.status_code, beautify_json(response.json())))
        except requests.exceptions.ConnectionError:
            click.echo('Error connecting with Blackbox API')


@entity.command('get_entity')
@click.argument('entity_id', default='')
@click.pass_context
def get_entities(ctx, entity_id):
    """
    Get the entities from Orion Context Broker (FIWARE) and Blackbox API

    Args:
        ctx (object): click object
        entity_id (str): Orion Context Broker (FIWARE) entity id
    """
    url = ctx.obj['FIWARE_HOST'] + '/v2/entities/' + entity_id
    headers = {'fiware-service': ctx.obj['FIWARE_SERVICE'], 'fiware-servicepath': '/'}
    try:
        response = requests.get(url=url, headers=headers)
        click.echo('[FIWARE] STATUS_CODE: {}, JSON: {}'.format(response.status_code, beautify_json(response.json())))
    except requests.exceptions.ConnectionError:
        click.echo('Error connecting with Orion Context Broker')

    url = ctx.obj['BLACKBOX_HOST'] + '/api/v1/anomaly/entity/' + entity_id
    try:
        response = requests.get(url=url)
        click.echo(
            '[BLACKBOX API] STATUS_CODE: {}, MSG: {}'.format(response.status_code, beautify_json(response.json())))
    except requests.exceptions.ConnectionError:
        click.echo('Error connecting with Blackbox API')


#################
# SUBSCRIPTIONS #
#################

@entity.command('create_subs')
@click.argument('entity_id')
@click.argument('url')
@click.argument('attrs', nargs=-1)
def create_subs(entity_id, url, attrs):
    """
    Create a subscription for an entity in Orion Context Broker.

    Args:
        entity_id (str): Orion Context Broker (FIWARE) entity id
        url (str): URL where Orion Context Broker will make an HTTP POST.
        attrs (list of str): names of the attrs that will be send when there is a change in its value.
    """
    click.echo(entity_id)
    click.echo(url)
    click.echo(attrs)


@entity.command('delete_subs')
@click.argument('subscription_id')
def create_subs(subscription_id):
    click.echo(subscription_id)


@entity.command('get_subs')
@click.argument('subscription_id', default='')
def get_subs(subscription_id):
    click.echo(subscription_id)


if __name__ == '__main__':
    entity()
