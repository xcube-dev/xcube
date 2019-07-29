import os


def get_inputdata_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), 'inputdata', name)


def create_input_txt(range_array):
    f = open((os.path.join(os.path.dirname(__file__), 'inputdata', "input.txt")), "w+")
    for i in range_array:
        file_name = f"2017010{i}-IFR-L4_GHRSST-SSTfnd-ODYSSEA-NWE_002-v2.0-fv1.0.nc"
        path = get_inputdata_path(file_name)
        f.write("%s\n" % path)
    f.close()
    return
