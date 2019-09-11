#!/usr/bin/env python3

import click
import requests

DEFAULT_FIWARE_HOST = 'http://localhost:1026'
DEFAULT_BLACKBOX_HOST = 'http://localhost:5678'
DEFAULT_SERVICE_PATH = 'sensores'
DEFAULT_TYPE = 'Machine'


@click.group()
@click.option('--entity_id', '-id', required=True, help='Entity ID (Orion Context Broker)')
@click.option('--on', type=click.Choice(['fiware', 'blackbox', 'both']), default='both',
              help='Indicates where the operation has to be done.')
@click.option('--blackbox_url', '-ah', default=DEFAULT_BLACKBOX_HOST, help='URL of Blackbox API')
@click.option('--fiware_url', '-fh', default=DEFAULT_FIWARE_HOST, help='URL of Orion Context Broker (FIWARE)')
@click.option('--service_path', '-sp', default=DEFAULT_SERVICE_PATH, help='Orion Context Broker service path')
@click.option('--entity_type', '-et', default=DEFAULT_TYPE, help='Type of the entity')
@click.pass_context
def entity(ctx, entity_id, on, blackbox_url, fiware_url, service_path, entity_type):
    ctx.ensure_object(dict)
    ctx.obj['ENTITY_ID'] = entity_id
    ctx.obj['ON'] = on
    ctx.obj['BLACKBOX_HOST'] = blackbox_url
    ctx.obj['FIWARE_HOST'] = fiware_url
    ctx.obj['SERVICE_PATH'] = service_path
    ctx.obj['TYPE'] = entity_type

@entity.command('create')
@click.pass_context
def create_entity(ctx):
    """
    Creates an entity in Orion Context Broker (FIWARE) and in the Blackbox Anomaly Detection Model.

    Args:
        ctx (object): click object
    """
    click.echo('Creating {} in {} & {} (service: {})'.format(ctx.obj['ENTITY_ID'], ctx.obj['BLACKBOX_HOST'],
                                                             ctx.obj['FIWARE_HOST'], ctx.obj['SERVICE_PATH']))

    if ctx.obj['ON'] == 'fiware' or ctx.obj['ON'] == 'both':
        # create entity in Orion Context Broker
        url = ctx.obj['FIWARE_HOST'] + '/v2/entities'
        data = {
            'id': ctx.obj['ENTITY_ID'],
            'type': ctx.obj['TYPE']
        }
        headers = {
            'Content-Type': 'application/json',
            'fiware-service': ctx.obj['SERVICE_PATH'],
            'fiware-servicepath': '/',
        }
        response = requests.post(url=url, headers=headers, json=data)
        click.echo('[FIWARE] STATUS_CODE: {}'.format(response.status_code))

    if ctx.obj['ON'] == 'blackbox' or ctx.obj['ON'] == 'both':
        # create entity in Blackbox API
        url = ctx.obj['BLACKBOX_HOST'] + '/api/v1/anomaly/entity/' + ctx.obj['ENTITY_ID']
        response = requests.post(url=url, json={'attrs': []})
        click.echo('[BLACKBOX API] STATUS_CODE: {}, MSG: {}'.format(response.status_code, response.json()))


@entity.command('delete')
@click.pass_context
def delete_entity(ctx):
    """
    Deletes an entity in Orion Context Broker (FIWARE) and in the Blackbox Anomaly Detection Model.

    Args:
        ctx (object): click object
    """
    click.echo('Deleting {} in {} & {} (service: {})'.format(ctx.obj['ENTITY_ID'], ctx.obj['BLACKBOX_HOST'],
                                                             ctx.obj['FIWARE_HOST'], ctx.obj['SERVICE_PATH']))

    if ctx.obj['ON'] == 'fiware' or ctx.obj['ON'] == 'both':
        # deletes an entity in Orion Context Broker
        url = ctx.obj['FIWARE_HOST'] + '/v2/entities/' + ctx.obj['ENTITY_ID']
        headers = {'fiware-service': ctx.obj['SERVICE_PATH'], 'fiware-servicepath': '/'}
        response = requests.delete(url=url, headers=headers)
        click.echo('[FIWARE] STATUS_CODE: {}'.format(response.status_code))

    if ctx.obj['ON'] == 'blackbox' or ctx.obj['ON'] == 'both':
        # deletes an entity in Blackbox API
        url = ctx.obj['BLACKBOX_HOST'] + '/api/v1/anomaly/entity/' + ctx.obj['ENTITY_ID']
        response = requests.delete(url=url)
        click.echo('[BLACKBOX API] STATUS_CODE: {}, MSG: {}'.format(response.status_code, response.json()))


@entity.command('add')
@click.argument('attrs', nargs=-1)
@click.pass_context
def add(ctx, attrs):
    """
    Adds the passed attributes to an entity in Orion Context Broker (FIWARE) and Blackbox API. i.e: "Bearing1,Float,0.67"

    Args:
        ctx (object): click object
        attrs (list of str): list of strings containing the name, type and value of the attributes.
    """

    # parse attributes
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

    if ctx.obj['ON'] == 'fiware' or ctx.obj['ON'] == 'both':
        # add attributes to Orion Context Broker
        url = ctx.obj['FIWARE_HOST'] + '/v2/entities/' + ctx.obj['ENTITY_ID'] + '/attrs'
        headers = {
            'Content-Type': 'application/json',
            'fiware-service': ctx.obj['SERVICE_PATH'],
            'fiware-servicepath': '/'
        }
        response = requests.post(url=url, headers=headers, json=attr_dict)
        click.echo('[FIWARE] STATUS_CODE: {}'.format(response.status_code))

    if ctx.obj['ON'] == 'blackbox' or ctx.obj['ON'] == 'both':
        # add attributes to Blackbox API
        url = ctx.obj['BLACKBOX_HOST'] + '/api/v1/anomaly/entity/' + ctx.obj['ENTITY_ID']
        attrs = [key for key in attr_dict.keys()]
        response = requests.put(url=url, json={'attrs': attrs})
        click.echo('[BLACKBOX API] STATUS_CODE: {}, MSG: {}'.format(response.status_code, response.json()))


if __name__ == '__main__':
    entity()
