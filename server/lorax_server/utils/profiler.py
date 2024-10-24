import time
from contextlib import contextmanager

import torch


class TimingContextManager:
    def __init__(self, name: str):
        self.name = name
        self.total_time = 0
        self.count = 0

    @contextmanager
    def timing(self):
        start = time.time()
        try:
            yield
        finally:
            end = time.time()
            self.total_time += end - start
            self.count += 1
            # print(f"=== {self.name}: avg={self.get_average_time():.3f} s  total={self.total_time:.3f} s  count={self.count}") 

    def get_average_time(self):
        if self.count == 0:
            return 0
        return self.total_time / self.count


_timers = {}


@contextmanager
def timer(name: str):
    if name not in _timers:
        _timers[name] = TimingContextManager(name)
    with _timers[name].timing():
        yield
        # torch.cuda.synchronize()
