# The MIT License (MIT)
# Copyright (c) 2019 by the xcube development team and contributors
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

# Force loading of xarray
# noinspection PyUnresolvedReferences
from .api import XCubeDatasetAccessor
from .compute import compute_cube
from .dump import dump_dataset
from .evaluate import evaluate_dataset
from .extract import DEFAULT_INDEX_NAME_PATTERN, DEFAULT_INTERP_POINT_METHOD, DEFAULT_REF_NAME_PATTERN, \
    get_cube_point_indexes, get_cube_values_for_indexes, get_cube_values_for_points, get_dataset_indexes
from .gen.gen import gen_cube
from .levels import compute_levels, read_levels, write_levels
from .new import new_cube
from .readwrite import open_cube, open_dataset, read_cube, read_dataset, write_cube, write_dataset
from .resample import resample_in_time
from .select import select_vars
from .ts import get_time_series
from .vars_to_dim import vars_to_dim
from .verify import assert_cube, verify_cube
# noinspection PyUnresolvedReferences
from ..util.chunk import chunk_dataset, get_empty_dataset_chunks
# noinspection PyUnresolvedReferences
from ..util.cubestore import CubeStore
# noinspection PyUnresolvedReferences
from ..util.edit import edit_metadata
# noinspection PyUnresolvedReferences
from ..util.geom import clip_dataset_by_geometry, convert_geometry, mask_dataset_by_geometry
# noinspection PyUnresolvedReferences
from ..util.maskset import MaskSet
# noinspection PyUnresolvedReferences
from ..util.optimize import optimize_dataset
# noinspection PyUnresolvedReferences
from ..util.schema import CubeSchema
# noinspection PyUnresolvedReferences
from ..util.unchunk import unchunk_dataset
# noinspection PyUnresolvedReferences
from ..util.update import update_dataset_attrs, update_dataset_spatial_attrs, update_dataset_temporal_attrs, \
    update_dataset_var_attrs
