"""
A simple processing service compatible with the actual xcube Generator
REST API (in xcube-hub).

It generates data cubes by invoking "xcube gen2" CLI as local processes,
see generate_cube().

It creates and uses a simple data store named "test" that contains two
datasets: DATASET-1.zarr and DATASET-2.zarr, see _init_local_store().
"""

import datetime
import json
import os
import os.path
import subprocess
from typing import Dict, Any, Tuple, Optional

import click
import flask
import werkzeug.exceptions
import yaml

from xcube.util.temp import new_temp_dir
from xcube.util.temp import new_temp_file

app = flask.Flask(__name__)

STORES_CONFIG_PATH: str = ''
JOBS: Dict[str, Tuple[subprocess.Popen, str, Dict[str, Any]]] = {}


@app.route('/status')
def status():
    return {'status': 'ok'}


@app.route('/cubegens', methods=['PUT'])
def generate_cube():
    request_path = _new_request_file(flask.request)
    try:
        process = subprocess.Popen(['xcube', 'gen2',
                                    '-vvv',
                                    '--stores', STORES_CONFIG_PATH,
                                    request_path],
                                   stderr=subprocess.STDOUT)
        return _process_to_generator_result(process, None)
    except subprocess.CalledProcessError as e:
        raise werkzeug.exceptions.InternalServerError(
            'failed to invoke generator process'
        ) from e


@app.route('/cubegens/<job_id>', methods=['GET'])
def get_cube_generator_status(job_id: str):
    return _process_to_generator_result(None, job_id)


@app.route('/cubegens/info', methods=['POST'])
def get_cube_info():
    raise werkzeug.exceptions.NotImplemented()


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
        _, user_code_path = new_temp_file(suffix=user_code_storage.filename)
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
    _, request_path = new_temp_file(suffix='_request.yaml')
    with open(request_path, 'w') as stream:
        yaml.dump(request_dict, stream)

    print(f' * Request: {request_path}')
    return request_path


def _init_local_store():
    """
    Initialize a "directory" data store with test datasets.
    """
    from xcube.core.new import new_cube

    local_base_dir = new_temp_dir(suffix='_local_store')
    dataset_1 = new_cube(width=36, height=18, variables={'A': 0.1, 'B': 0.2})
    dataset_2 = new_cube(width=36, height=18, variables={'C': 0.2, 'D': 0.3})
    dataset_1.to_zarr(os.path.join(local_base_dir, 'DATASET-1.zarr'))
    dataset_2.to_zarr(os.path.join(local_base_dir, 'DATASET-2.zarr'))

    global STORES_CONFIG_PATH
    _, STORES_CONFIG_PATH = new_temp_file(suffix='_stores.yaml')
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


def _process_to_generator_result(process: Optional[subprocess.Popen],
                                 job_id: Optional[str]) -> Dict[str, Any]:
    global JOBS
    if process is not None:
        job_id = str(process.pid)
        start_time = datetime.datetime.utcnow().isoformat()
        result = {}
        JOBS[job_id] = process, start_time, result
    elif job_id in JOBS:
        process, start_time, result = JOBS[job_id]
    else:
        raise werkzeug.exceptions.BadRequest(
            f'invalid job ID: {job_id}'
        )
    output = process.stdout.readlines() \
        if process.stdout is not None else None
    return_code = process.poll()
    active, succeeded, failed = None, None, None
    if return_code is None:
        active = 1
    else:
        succeeded = 1 if return_code == 0 else 0
        failed = 1 if return_code != 0 else 0
    result.update({
        'cubegen_id': job_id,
        'status': {
            'active': active,
            'succeeded': succeeded,
            'failed': failed,
            'start_time': start_time,
        },
        'output': output,
    })
    return result


@click.command()
@click.option('--host', '-h', help='Host address.', default="127.0.0.1")
@click.option('--port', '-p', help='Port number.', default=5000, type=int)
def server(host, port):
    """Test server for xcube generator."""
    _init_local_store()
    app.run(host=host, port=port)


if __name__ == '__main__':
    server()
