import numpy as np

"""agent class"""


class Agent:
    def __init__(
        self,
        init_position: tuple[int],
        init_fitness: float,
        id=int,
    ):
        self.position_hist = [init_position]
        self.fitness_hist = [init_fitness]
        self.Pi_hist = []
        self.id = id
        self._Pi = None

        self.fights_won = 0
        self.fights_lost = 0
        self.Xi_hist = []
        self._Xi = None

    @property
    def position(self):
        return self.position_hist[-1]

    @property
    def fitness(self):
        return self.fitness_hist[-1]

    @property
    def Pi(self):
        return self._Pi

    @Pi.setter
    def Pi(self, value):
        self._Pi = value
        self.Pi_hist.append(value)

    @property
    def Xi(self):
        return self._Xi

    @Xi.setter
    def Xi(self, value):
        self._Xi = value
        self.Xi_hist.append(value)

    def add_unchanged_step(self):
        self.fitness_hist.append(self.fitness)

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
