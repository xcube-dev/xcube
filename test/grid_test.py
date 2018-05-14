import math
import unittest


class GridSpec:

    def __init__(self, cell_size, chunk_size, num_level_zero_chunks):
        self.cell_size = _to_xy_tuple(cell_size, float)
        self.chunk_size = _to_xy_tuple(chunk_size, int)
        self.num_level_zero_chunks = _to_xy_tuple(num_level_zero_chunks, int)


def _to_xy_tuple(v, t):
    try:
        x, y = v
    except ValueError:
        x, y = v, v
    return t(x), t(y)


def find_res(grid_extent: float, cell_extent_target: float, chunk_size_min=180):
    best_delta = float('inf')
    best_num_level_zero_chunks = -1
    best_chunk_size = -1
    best_num_levels = -1

    def compute_cell_extent(grid_extent, num_level_zero_chunks, chunk_size, level):
        grid_size = chunk_size * num_level_zero_chunks * (2 ** level)
        return grid_extent / grid_size, grid_size

    num_level_zero_chunks = 1
    while True:
        cell_extend, grid_size = compute_cell_extent(grid_extent, num_level_zero_chunks, 1, 0)
        break_num_level_zero_chunks = cell_extend < cell_extent_target
        chunk_size = min(chunk_size_min, grid_size)
        while True:
            cell_extend, grid_size = compute_cell_extent(grid_extent, num_level_zero_chunks, chunk_size, 0)
            break_chunk_size = cell_extend < cell_extent_target
            level = 0
            while True:
                cell_extend, grid_size = compute_cell_extent(grid_extent, num_level_zero_chunks, chunk_size, level)
                break_level = cell_extend < cell_extent_target
                delta = abs(cell_extend - cell_extent_target)
                if delta < best_delta and chunk_size >= chunk_size_min:
                    best_delta = delta
                    best_num_levels = level + 1
                    best_chunk_size = chunk_size
                    best_num_level_zero_chunks = num_level_zero_chunks
                if break_level:
                    break
                level += 1
            if break_chunk_size:
                break
            chunk_size += 1
        if break_num_level_zero_chunks:
            break
        num_level_zero_chunks += 1
    return best_delta, best_num_levels, best_chunk_size, best_num_level_zero_chunks


inf = float('inf')


class GridSpecTest(unittest.TestCase):
    def test_find_res(self):
        pass
        # self.assertEqual(find_res(360., 1.), (0.0, 2, 180, 1))
        # self.assertEqual(find_res(180., 1.), (0.0, 1, 180, 1))
        #
        # self.assertEqual(find_res(360., 180. / 100), (0.0, 1, 200, 1))
        # self.assertEqual(find_res(180., 180. / 100), (inf, -1, -1, -1))
        #
        # self.assertEqual(find_res(360., 180. / 1000), (0.0, 4, 250, 1))
        # self.assertEqual(find_res(180., 180. / 1000), (0.0, 3, 250, 1))
        #
        #
        # r = 6378137
        # self.assertEqual(find_res(360., 300 / r), (0.0, 4, 250, 1))
        # self.assertEqual(find_res(180., 300 / r), (0.0, 3, 250, 1))
