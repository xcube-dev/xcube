import atexit
import json
import os
import os.path
import shutil
import subprocess
import tempfile
import datetime
from typing import Dict

import click
import flask
import werkzeug.exceptions
import yaml

app = flask.Flask(__name__)

STORES_CONFIG_PATH: str = ''
JOBS: Dict[int, subprocess.Popen] = {}


@app.route('/status')
def status():
    return {'status': 'ok'}


@app.route('/cubegens/<job_id>', methods=['GET'])
def generate(job_id: str):
    try:
        pid = int(job_id)
    except ValueError:
        raise werkzeug.exceptions.BadRequest('invalid job ID')
    process = JOBS.get(pid)
    if process is None:
        raise werkzeug.exceptions.BadRequest('invalid job ID')
    output = process.stdout.readlines()


@app.route('/cubegens/info', methods=['POST'])
def generate(job_id: str):
    try:
        pid = int(job_id)
    except ValueError:
        raise werkzeug.exceptions.BadRequest('invalid job ID')
    process = JOBS.get(pid)
    if process is None:
        raise werkzeug.exceptions.BadRequest('invalid job ID')
    output = process.stdout.readlines()


@app.route('/cubegens', methods=['PUT'])
def generate():
    request_path = _new_request_file(flask.request)

    try:
        process = subprocess.Popen(['xcube', 'gen2',
                                    '-vvv',
                                    '--stores', STORES_CONFIG_PATH,
                                    request_path],
                                   stderr=subprocess.STDOUT)
        global JOBS
        JOBS[process.pid] = process
    except subprocess.CalledProcessError as e:
        raise werkzeug.exceptions.InternalServerError(
            'failed to invoke generator process'
        ) from e

    output = output.decode('utf-8') if output is not None else ''

    return {
        'cubegen_id': process.pid,
        'status': {
            'active': True,
            'start_time': datetime.datetime.utcnow().isoformat(),
        },
        'output': process.stdout.readlines(),

    }


def _new_request_file(request: flask.Request) -> str:
    """
    Process an incoming Flask request and
    return path to temporary request file.
    """

    request_files = request.files
    if request_files:
        # Multipart PUT request:
        # We expect two files: a JSON file "body"
        # and a binary file "user_code".

        # "body" contains the generator request and
        request_storage = request_files.get('body')
        request_dict = json.load(request_storage.stream)

        # "user_code" is a ZIP archive with the user code.
        user_code_storage = request_files.get('user_code')
        user_code_path = _new_temp_file(suffix=user_code_storage.filename)
        user_code_storage.save(user_code_path)
        print(f' * User code file: {user_code_path}')
    else:
        # Not a multipart PUT request: Expect JSON request in
        # body content:
        request_dict = request.json
        user_code_path = None

    if not isinstance(request_dict, dict):
        print(f'Error: received invalid request: {request_dict}')
        raise werkzeug.exceptions.BadRequest('request data must be JSON')

    if user_code_path:
        # If we have user code, alter the "code_config"
        # part of the generator request so it points to
        # the temporary ZIP archive.
        code_config = request_dict.get('code_config')
        if isinstance(code_config, dict):
            file_set = code_config.get('file_set')
            if isinstance(file_set, dict):
                file_set['path'] = user_code_path

    # Write the request to a temporary file
    request_path = _new_temp_file(suffix='_request.yaml')
    with open(request_path, 'w') as stream:
        yaml.dump(request_dict, stream)

    print(f' * Request: {request_path}')
    return request_path


def _init_local_store():
    """
    Initialize a "directory" data store with test datasets.
    """
    from xcube.core.new import new_cube

    local_base_dir = _new_temp_dir(suffix='_local_store')
    dataset_1 = new_cube(width=36, height=18, variables={'A': 0.1, 'B': 0.2})
    dataset_2 = new_cube(width=36, height=18, variables={'C': 0.2, 'D': 0.3})
    dataset_1.to_zarr(os.path.join(local_base_dir, 'DATASET-1.zarr'))
    dataset_2.to_zarr(os.path.join(local_base_dir, 'DATASET-2.zarr'))

    global STORES_CONFIG_PATH
    STORES_CONFIG_PATH = _new_temp_file(suffix='_stores.yaml')
    with open(STORES_CONFIG_PATH, 'w') as stream:
        yaml.dump(
            {
                'test': {
                    'title': 'Local test store',
                    'store_id': 'directory',
                    'store_params': {
                        'base_dir': local_base_dir
                    }
                }
            },
            stream)
    print(f' * Store base directory: {local_base_dir}')
    print(f' * Store configuration: {STORES_CONFIG_PATH}')


def _new_temp_file(suffix=''):
    return _register_for_removal(
        tempfile.mktemp(prefix='xcube-gen2-testserver_', suffix=suffix)
    )


def _new_temp_dir(suffix=''):
    return _register_for_removal(
        tempfile.mkdtemp(prefix='xcube-gen2-testserver_', suffix=suffix)
    )


def _register_for_removal(path: str) -> str:
    atexit.register(_remove_path, path)
    return path


def _remove_path(path: str):
    # noinspection PyBroadException
    try:
        print(f' * Removing {path}')
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
    except BaseException:
        pass


@click.command()
@click.option('--host', '-h', help='Host address.', default="127.0.0.1")
@click.option('--port', '-p', help='Port number.', default=5000, type=int)
def server(host, port):
    """Test server for xcube generator."""
    _init_local_store()
    app.run(host=host, port=port)


if __name__ == '__main__':
    server()
