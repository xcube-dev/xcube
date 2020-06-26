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

from xcube.cli._gen2.genconfig import GenConfig
from xcube.cli._gen2.open import open_cubes
from xcube.cli._gen2.progress import ApiProgressCallbackObserver
from xcube.cli._gen2.progress import ConsoleProgressObserver
from xcube.cli._gen2.resample import resample_and_merge_cubes
from xcube.cli._gen2.storeconfig import load_data_store_instances
from xcube.cli._gen2.write import write_cube
from xcube.util.progress import observe_progress


def main(gen_config_path: str,
         store_configs_path: str = None,
         verbose: bool = False):
    """
    Generator tool for data cubes.

    Creates cube views from one or more cube stores, resamples them to a common grid,
    optionally performs some cube transformation,
    and writes the resulting cube to some target cube store.

    *gen_config_path* is the cube generator configuration. It may be provided as a JSON or YAML file
    (file extensions ".json" or ".yaml"). If the *gen_config_path* argument is omitted, it is expected that
    the cube generator configuration is piped as a JSON string.

    *store_configs_path* is a path to a JSON file with data store configurations. It is a mapping of names to
    configured stores. Entries are dictionaries that have a mandatory "store_id" property which is a name of a
    registered xcube data store. The optional "store_params" property may define data store specific parameters.

    :param gen_config_path: Cube generation configuration. It may be provided as a JSON or YAML file
        (file extensions ".json" or ".yaml"). If the REQUEST file argument is omitted, it is expected that
        the cube generator configuration is piped as a JSON string.
    :param store_configs_path: A JSON file that maps store names to parameterized stores.
    :param verbose: Whether to output progress information to stdout.
    """

    store_instances = load_data_store_instances(store_configs_path)

    gen_config = GenConfig.from_file(gen_config_path, verbose=verbose)

    if gen_config.callback_config:
        ApiProgressCallbackObserver(gen_config.callback_config).activate()
    if verbose:
        ConsoleProgressObserver().activate()

    with observe_progress('Generating cube', 100) as cm:
        cm.will_work(10)
        cubes = open_cubes(gen_config.input_configs,
                           cube_config=gen_config.cube_config,
                           store_instances=store_instances)

        cm.will_work(10)
        cube = resample_and_merge_cubes(cubes,
                                        cube_config=gen_config.cube_config)

        cm.will_work(80)
        write_cube(cube,
                   output_config=gen_config.output_config,
                   store_instances=store_instances)
