import random


def gen_rows(n_geoms=3,
             n_times=4,
             n_depths=3):
    yield "GEOMETRY", "TIME", "DEPTH", "CHL", "TSM", "YS"
    for i_geom in range(n_geoms):
        for i_time in range(n_times):
            for i_depth in range(n_depths):
                yield (i_geom, i_time, i_depth,
                       random.randint(0, 10),
                       random.randint(0, 10),
                       random.randint(0, 10))


with open("vecna-cube.csv", mode="w") as file:
    for row in gen_rows():
        file.write(",".join(map(str, row)) + "\n")
