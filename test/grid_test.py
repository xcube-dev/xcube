import math
import unittest


def get_tile_counts_and_sizes(res: float, tile_size_max=720):
    tile_size_min = tile_size_max // 2 + 1

    height = int(round(180 / res))
    if height <= tile_size_min:
        return [(1, height)]

    tile_counts = []
    for tile_size in reversed(range(tile_size_min, tile_size_max + 1)):
        tile_count = height // tile_size
        if tile_count * tile_size == height:
            tile_counts.append((tile_count, tile_size))

    if not tile_counts:
        return [(1, height)]

    return tile_counts


class GetTileCountTest(unittest.TestCase):

    def test_tile_counts_and_sizes(self):
        # TODO: select tile_count from tile_counts_and_sizes list using following criteria:
        #       1. tile_size as large as possible
        #       2. tile_count with highest divisibility by 2
        for res in [1, 1/2, 1/4, 1/5, 1/8,
                    1/10, 1/20, 1/25, 1/50,
                    1/100, 1/200, 1/250, 1/500,
                    1/1000, 1/2000, 1/2500, 1/5000,
                    1/10000]:
            tc = get_tile_counts_and_sizes(res)
            print(res, tc)
