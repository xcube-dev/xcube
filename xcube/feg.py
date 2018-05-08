import math



class FixedEarthGrid:

    EARTH_EQUATORIAL_RADIUS = 63781370.

    def __init__(self, level_zero_res=1.0):
        self.level_zero_res = level_zero_res
        level_zero_grid_size_x = 360. / level_zero_res
        level_zero_grid_size_y = 180. / level_zero_res
        msg = '%s divided by level_zero_res must be an integer division'
        if level_zero_grid_size_x * level_zero_res != 360.:
            raise ValueError(msg % 360)
        if level_zero_grid_size_y * level_zero_res != 180.:
            raise ValueError(msg % 180)
        self.level_zero_grid_size_x = int(level_zero_grid_size_x)
        self.level_zero_grid_size_y = int(level_zero_grid_size_y)

    def get_res(self, level: int, units='degrees'):
        return self.from_degree(self.level_zero_res / (2 ** level), units=units)

    def get_level(self, res: float, units='degrees'):
        res = self.to_degree(res, units)
        level = int(round(math.log2(self.level_zero_res / res)))
        return level if level >= 0 else 0

    def get_level_and_res(self, res: float, units='degrees'):
        level = self.get_level(res, units=units)
        return level, self.from_degree(self.get_res(level), units=units)

    def get_grid_size(self, level):
        scale = 2 ** level
        return self.level_zero_grid_size_x * scale, self.level_zero_grid_size_y * scale

    @classmethod
    def to_degree(cls, res, units):
        if units == 'degrees':
            return res
        if units == 'meters':
            return (360.0 * res) / cls.EARTH_EQUATORIAL_RADIUS
        raise ValueError(f'unrecognized units {units!r}')

    @classmethod
    def from_degree(cls, res, units):
        if units == 'degrees':
            return res
        if units == 'meters':
            return (res / 360.0) * cls.EARTH_EQUATORIAL_RADIUS
        raise ValueError(f'unrecognized units {units!r}')
