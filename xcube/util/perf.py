import time


class perf_measure:
    def __init__(self, output=False):
        self.output = output

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delta = time.perf_counter() - self.t0
        if self.output:
            print(self.delta, 'seconds')
