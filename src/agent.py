"""agent class"""

import numpy as np


class Agent:
    def __init__(
        self,
        group: int,
        init_position: tuple[int],
        init_fitness: float,
        id=int,
    ):
        self.group = group
        self.position_hist = [init_position]
        self.fitness_hist = [init_fitness]
        self.id = id

    @property
    def position(self):
        return self.position_hist[-1]

    @property
    def fitness(self):
        return self.fitness_hist[-1]

    def add_unchanged_step(self):
        pos, fit = self.position, self.fitness
        self.position_hist.append(pos)
        self.fitness_hist.append(fit)

    def move_position(self, L):
        x, y = self.position

        move_axis = np.random.choice([0, 1])

        if move_axis == 0:
            dx = np.random.choice([-1, 1])
            x = int(np.clip(x + dx, 0, L - 1))
        else:
            dy = np.random.choice([-1, 1])
            y = int(np.clip(y + dy, 0, L - 1))
        self.position_hist.append((x, y))
