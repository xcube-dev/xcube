import math

# WGS84 ellipsoid semi-major axis
WGS84_ELLIPSOID_SEMI_MAJOR_AXIS = 6378137.
EARTH_EQUATORIAL_PERIMETER = 2. * math.pi * WGS84_ELLIPSOID_SEMI_MAJOR_AXIS

SEP = ";"
T_MIN = 180
L_MAX = 15


def degrees_to_meters(res):
    return (res / 360.0) * EARTH_EQUATORIAL_PERIMETER


def res_table():
    table = []

    header = ["T", "N", "1/res0 (1/deg)", "res0 (deg)"]
    for i in range(0, L_MAX + 1):
        header.append(f"res{i}")
    table.append(tuple(header))

    for t in range(T_MIN, 25 * T_MIN + 1):
        if t > T_MIN and t % T_MIN == 0:
            continue
        for n in range(0, L_MAX + 1):
            u = t * 2 ** n
            m = u // T_MIN
            if T_MIN * m == u:
                row = [t, n, m, 1 / m]
                res0 = degrees_to_meters(1 / m)
                for i in range(0, L_MAX + 1):
                    row.append(res0 * math.pow(2, -i))
                table.append(tuple(row))
                break
    return tuple(table)


if __name__ == "__main__":
    table = res_table()
    with open("grid-resolutions.csv", "w") as fp:
        fp.writelines([SEP.join(map(str, row)) + "\n" for row in table])
