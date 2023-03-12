from contextlib import contextmanager
import time


@contextmanager
def track_process_time(arr):
    start = time.time()
    yield
    end = time.time()

    arr.append(end - start)
