"""
A simple processing remote compatible with the actual xcube Generator
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
import jsonschema
import werkzeug.exceptions
import yaml

from xcube.core.gen2.request import CubeGeneratorRequest
from xcube.util.temp import new_temp_dir
from xcube.util.temp import new_temp_file

Job = Tuple[
    subprocess.Popen,  # process
    str,  # result_path
    str,  # start_time
    Dict[str, Any],  # updated state object
]

STORES_CONFIG_PATH: str = ''
JOBS: Dict[str, Job] = {}

app = flask.Flask('test.core.gen2.remote.server')


@click.command()
@click.option('--host', '-h', help='Host address.', default="127.0.0.1")
@click.option('--port', '-p', help='Port number.', default=5000, type=int)
def server(host, port):
    """Test server that emulates the xcube generator service."""
    _init_local_store()
    app.run(host=host, port=port, debug=True)


@app.route('/status')
def status():
    return {'status': 'ok'}


@app.route('/cubegens/info', methods=['POST'])
def get_cube_info():
    """Get information about cube to be generated"""
    return _get_cube_info_result()


@app.route('/cubegens', methods=['PUT'])
def generate_cube():
    """Normal /cubegens where body = generator request JSON"""
    return _generate_cube()


@app.route('/cubegens/code', methods=['PUT'])
def generate_cube_from_code():
    """Special variant of /cubegens that uses a multipart request"""
    return _generate_cube()


@app.route('/cubegens/<job_id>', methods=['GET'])
def get_cube_generator_status(job_id: str):
    """Get status info for cube generator job <job_id>"""
    return _get_generate_cube_result(None, None, job_id)


def _get_cube_info_result():
    request_path = _new_request_file(flask.request)
    result_path = _new_result_file()
    args = ['xcube', 'gen2',
            '--info',
            '--stores', STORES_CONFIG_PATH,
            '--output', result_path,
            request_path]

    try:
        output = subprocess.check_output(args)
    except subprocess.CalledProcessError as e:
        raise werkzeug.exceptions.InternalServerError(
            f'failed to invoke generator process: {args!r}'
        ) from e

    with open(result_path) as fp:
        result_json = json.load(fp)

    if isinstance(output, bytes):
        text = output.decode('utf-8')
        ansi_suffix = '\x1b[0m'
        if text.endswith(ansi_suffix):
            # remove ANSI escape sequence (on Windows only?)
            text = text[0:-len(ansi_suffix)]
        if text:
            result_json.update(output=text.split('\n'))

    status_code = _get_set_status_code(result_json)

    return result_json, status_code


def _generate_cube():
    request_path = _new_request_file(flask.request)
    result_path = _new_result_file()
    args = ['xcube', 'gen2',
            '-vv',
            '--stores', STORES_CONFIG_PATH,
            '--output', result_path,
            request_path]

    try:
        process = subprocess.Popen(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise werkzeug.exceptions.InternalServerError(
            f'failed to invoke generator process: {args!r}'
        ) from e

    return _get_generate_cube_result(process, result_path, None)


def _get_generate_cube_result(
        process: Optional[subprocess.Popen],
        result_path: Optional[str],
        job_id: Optional[str]
) -> Tuple[Dict[str, Any], int]:
    global JOBS

    if process is not None:
        job_id = str(process.pid)
        start_time = datetime.datetime.utcnow().isoformat()
        state = {}
        JOBS[job_id] = process, result_path, start_time, state
    elif job_id in JOBS:
        process, result_path, start_time, state = JOBS[job_id]
    else:
        raise werkzeug.exceptions.BadRequest(
            f'invalid job ID: {job_id}'
        )

    output = process.stdout.readlines() \
        if process.stdout is not None else None
    # print(type(output), output)

    return_code = process.poll()

    active, succeeded, failed = None, None, None
    job_result = None
    if return_code is None:
        active = 1
    else:
        if os.path.exists(result_path):
            with open(result_path) as fp:
                job_result = json.load(fp)
        if return_code == 0:
            succeeded = 1
        else:
            failed = 1

    state.update({
        'job_id': job_id,
        'job_status': {
            'active': active,
            'succeeded': succeeded,
            'failed': failed,
            'start_time': start_time,
        },
    })

    if output is not None:
        state.update(output=output)

    if job_result is not None:
        status_code = _get_set_status_code(job_result, success_code=201)
        state.update(job_result=job_result)
    else:
        status_code = 200 if succeeded else 400

    return state, status_code


def _get_set_status_code(result_json: Dict[str, Any],
                         success_code: int = 200,
                         failure_code: int = 400) -> int:
    status_code = result_json.get('status_code')
    if status_code is None:
        if result_json.get('status') == 'ok':
            status_code = success_code
        else:
            status_code = failure_code
        result_json['status_code'] = status_code
    return status_code


def _new_result_file() -> str:
    _, file_path = new_temp_file(prefix='xcube-gen2-', suffix='-result.json')
    return file_path


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

    try:
        CubeGeneratorRequest.get_schema().validate_instance(request_dict)
    except jsonschema.ValidationError as e:
        print(f'Error: received invalid request: {request_dict}')
        raise werkzeug.exceptions.BadRequest('request is no valid JSON') from e

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
                    'store_id': 'file',
                    'store_params': {
                        'root': local_base_dir
                    }
                }
            },
            stream)
    print(f' * Store base directory: {local_base_dir}')
    print(f' * Store configuration: {STORES_CONFIG_PATH}')


if __name__ == '__main__':
    server()
