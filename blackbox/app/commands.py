import os
import click
import json
from blackbox.app.extensions import api
from flask.cli import with_appcontext


@click.command(help="Export API as a Postman collection")
@click.option(
    "--path", default="./", help="Directory where to save the Postman collection",
)
@with_appcontext
def postman(path):
    """
    Export API as a Postman collection.

    Args:
        path (str): diretory where to save the Postman collection. Defaults to './'.
    """
    data = api.as_postman(urlvars=True, swagger=True)
    file_path = os.path.join(path, "blackbox_postman.json")

    with open(file_path, "w") as f:
        json.dump(data, f)
