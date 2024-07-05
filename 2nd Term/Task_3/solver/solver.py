from typing import NamedTuple
import numpy as np

class Interval(NamedTuple):
    start: float
    end: float
    step: float = 1e-2
    def get_axis(self):
        return np.arange(self.start, self.end, self.step)

class Space(NamedTuple):
    x: Interval
    t: Interval
    def get_meshgrid(self):
        return np.meshgrid(
            self.x.get_axis(), self.t.get_axis()
        )


class Sol():
    def __init__(self, space: Space, V: np.ndarray) -> None:
        self.space: Space = space
        self.dh: float = space.x.step
        self.dt: float = space.t.step
        self.axis_x, self.axis_t = self.space.get_meshgrid()
        self.map = np.zeros((self.axis_x.shape[1], self.axis_t.shape[0]), dtype=np.complex128)
        self.V = V

    def set_bounds(self, bound_values: np.ndarray):
        self.map = bound_values.copy()
        return self

    def set_initials(self, init_values: np.ndarray):
        self.map[:, 0] = init_values[:]
        return self

    def run(self):
        raise NotImplementedError
