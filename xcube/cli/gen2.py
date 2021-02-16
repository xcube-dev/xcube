# The MIT License (MIT)
# Copyright (c) 2020 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import click


@click.command(name="gen2", hidden=True)
@click.argument('gen_config_path', type=str, required=False, metavar='GEN_CONFIG')
@click.option('--store-conf', '-s', 'store_configs_path', metavar='STORE_CONFIGS',
              help='A JSON file that maps store names to parameterized stores.')
@click.option('--verbose', '-v',
              is_flag=True,
              help='Control amount of information dumped to stdout.')
def gen2(gen_config_path: str,
         store_configs_path: str = None,
         verbose: bool = False):
    """
    Generator tool for data cubes.

    Creates a cube view from one or more cube stores, optionally performs some cube transformation,
    and writes the resulting cube to some target cube store.

    GEN_CONFIG is the cube generation request. It may be provided as a JSON or YAML file
    (file extensions ".json" or ".yaml"). If the GEN_CONFIG file argument is omitted, it is expected that
    the Cube generation request is piped as a JSON string.

    STORE_CONFIGS is a path to a JSON file (file extensions ".json") with data store configurations.
    It is a mapping of arbitrary store names to configured data stores. Entries are dictionaries
    that have a mandatory "store_id" property which is a name of a registered xcube data store.
    The optional "store_params" property may define data store specific parameters.
    The following example defines a data store named "my_s3_store" which is an AWS S3 bucket store,
    and a data store "my_test_store" for testing, which is an in-memory data store:

    \b
    {
        "my_s3_store": {
            "store_id": "s3",
            "store_params": {
                "bucket_name": "eurodatacube",
                "aws_access_key_id": "jokljkjoiqqjvlaksd",
                "aws_secret_access_key": "1728349182734983248234"
            }
        },
        "my_test_store": {
            "store_id": "memory"
        }
    }

    """
    from xcube.core.gen2 import CubeGenerator
    from xcube.core.gen2 import CubeGeneratorError
    from xcube.core.store import DataStoreError
    try:
        CubeGenerator.from_file(gen_config_path,
                                store_configs_path=store_configs_path,
                                verbose=verbose).run()
    except (CubeGeneratorError, DataStoreError) as e:
        raise click.ClickException(f'{e}') from e


if __name__ == '__main__':
    gen2()
