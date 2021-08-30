import threading
from typing import Optional

import xarray as xr

# noinspection PyUnresolvedReferences
from xcube.core.evaluate import evaluate_dataset
# noinspection PyUnresolvedReferences
from xcube.core.gridmapping import GridMapping
from xcube.core.normalize import decode_cube, DatasetIsNotACubeError


@xr.register_dataset_accessor('xcube')
class XcubeAccessor:
    """
    Defines some new properties for datasets:

    * :attr:cube The subset of variables of this dataset
        which all have cube dimensions (time, ..., <y_name>, <x_name>).
        May be an empty dataset.
    * :attr:gm The grid mapping used by this dataset.
        It is an instance of :class:GridMapping.
        May be None, if this dataset does not define a grid mapping.

    :param dataset: An xarray dataset instance.
    """

    def __init__(self, dataset: xr.Dataset):
        self._dataset: xr.Dataset = dataset
        self._cube_subset: Optional[xr.Dataset] = None
        self._grid_mapping: Optional[GridMapping] = None
        self._lock = threading.RLock()

    @property
    def cube(self) -> xr.Dataset:
        if self._cube_subset is None:
            with self._lock:
                self._init_cube_subset()
        return self._cube_subset

    @property
    def gm(self) -> Optional[GridMapping]:
        if self._cube_subset is None:
            with self._lock:
                self._init_cube_subset()
        return self._grid_mapping

    def _init_cube_subset(self):
        try:
            cube, grid_mapping, _ = decode_cube(self._dataset,
                                                normalize=True,
                                                force_copy=True)
        except DatasetIsNotACubeError:
            cube, grid_mapping = xr.Dataset(), None
        self._cube_subset = cube
        self._grid_mapping = grid_mapping
