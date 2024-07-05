from typing import NamedTuple
import numpy as np

class Interval(NamedTuple):
    start: float
    end: float
class Sizes(NamedTuple):
    x: Interval
    y: Interval
class ISolver():
    def __init__(self, sizes: Sizes, dh: float, rtol: float) -> None:
        self.sizes: Sizes = sizes
        self.dh: float = dh
        self.rtol: float = rtol
        self.axis_x, self.axis_y = np.mgrid[
            	slice(sizes.x.start, sizes.x.end, dh),
                slice(sizes.y.start, sizes.y.end, dh)
            ]
        self.map = np.zeros((len(self.axis_x), len(self.axis_y)))

    def set_bounds(self, bound_map, bound_values) -> None:
        self.bound_map = bound_map
        self.bound_values = bound_values
        self.map = bound_values.copy()

    def run(self):
        raise NotImplementedError
