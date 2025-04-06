"""game class"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from src.agent import Agent
from src.utils import create_gif_from_images, get_time_string


class Game:
    def __init__(
        self,
        n_a: int,
        n_b: int,
        grid_size: int = 20,
        eta: float = 10,
        x: float = 0.05,
    ):
        self.groups = ["A", "B"]
        # input args
        self.n_a = n_a
        self.n_b = n_b
        self.grid_size = grid_size
        self.eta = eta
        self.x = x
        # deduced args
        self.n = n_a + n_b
        self.init_fitness = 1000 / self.n
        self.p_a = self.n_a / self.n
        self.p_b = self.n_b / self.n
        # to be modular in case we need to add a third group for example
        self.demography = {"A": self.n_a, "B": self.n_b}
        # population
        self.population = self.init_population()
        # min/max fitness
        self.extreme_fitness = self.init_extreme_fitness()
        # plot colors
        self.plot_colors = {"A": "Blues", "B": "Greens"}
        # saving path
        self.path = os.path.join("output", get_time_string())
        self.path_grid = os.path.join(self.path, "grid")
        os.mkdir(self.path)
        os.mkdir(self.path_grid)

    def play_game(
        self, t_max: int, save_every: int | None = None, plot: bool = False
    ):
        """Plays according to the diagram of the paper."""
        for t in tqdm.tqdm(range(t_max)):
            self.play_one_iter()
            # plot
            if save_every is not None and t % save_every == 0:
                self.plot_current_situation(t, plot=plot)
        # plot
        self.plot_fitness(t_max, plot=plot)
        if save_every is not None:
            create_gif_from_images(self.path_grid, self.path_grid)

    def play_one_iter(self):
        """Play one iteration of the game"""
        # move
        moving_group, moving_agent = self.move_agent()
        new_pos = np.array(self.population[moving_group][moving_agent].position)
        # checking for fight
        unmoving_group = "A" if moving_group == "B" else "B"
        position_unmov, _ = self.get_current_info_group(unmoving_group)
        fight_locations = np.argwhere(
            np.abs(new_pos - position_unmov).sum(axis=1) < 1e-4
        )
        # fight case
        unmoving_agent = self.check_and_do_fight(
            moving_group, moving_agent, unmoving_group, fight_locations
        )
        self.update_extreme_fitness()
        protagonists = {
            moving_group: moving_agent,
            unmoving_group: unmoving_agent,
        }
        self.update_non_active_agents(protagonists)

    def check_and_do_fight(
        self, moving_group, moving_agent, unmoving_group, fight_locations
    ):
        """Checks if a fight should take place and do it in such a case.
        Otherwise, the fitness is left unchanged and unmoving_agent
        is set to None to update it as the others."""
        # a fight is detected
        if fight_locations.size > 0:
            # chose a competitor among the unmoving group
            unmoving_agent = np.random.choice(fight_locations[0])
            self.fight(
                moving_group, moving_agent, unmoving_group, unmoving_agent
            )
        # nobody on the location
        else:
            fit = self.population[moving_group][moving_agent].fitness
            self.population[moving_group][moving_agent].fitness_hist.append(fit)
            unmoving_agent = None
        return unmoving_agent

    def update_non_active_agents(self, protagonists: dict):
        """All the agents that are not fighting are updated (left unchanged)."""
        for group, idx_group in protagonists.items():
            for idx_agent in self.population[group]:
                if idx_agent != idx_group:
                    self.population[group][idx_agent].add_unchanged_step()

    def move_agent(self):
        """Chose a group, an agent and moves it.
        Returns the info cause we need it."""
        moving_group = np.random.choice(self.groups, p=[self.p_a, self.p_b])
        moving_agent = np.random.choice(
            list(self.population[moving_group].keys())
        )
        self.population[moving_group][moving_agent].move_position(
            self.grid_size
        )
        return moving_group, moving_agent

    def fight(self, group_1, idx_1, group_2, idx_2):
        """Compute the winning probability and exchange
        the fitness between the fighters."""
        prob = self.get_probability(group_1, idx_1, group_2, idx_2)
        u = np.random.rand()
        fit_1 = self.population[group_1][idx_1].fitness
        fit_2 = self.population[group_2][idx_2].fitness

        if ((u < prob) and (group_1 == "A")) or (
            (u >= prob) and (group_1 == "B")
        ):
            self.population[group_1][idx_1].fitness_hist.append(
                fit_1 + self.x * fit_2
            )
            self.population[group_2][idx_2].fitness_hist.append(
                fit_2 - self.x * fit_2
            )
        else:
            self.population[group_1][idx_1].fitness_hist.append(
                fit_1 - self.x * fit_1
            )
            self.population[group_2][idx_2].fitness_hist.append(
                fit_2 + self.x * fit_1
            )

    def get_probability(self, group_1, idx_1, group_2, idx_2):
        """Compute the winning probability with normalized fitness"""
        f_hat_1 = self.get_normalized_fitness(group_1, idx_1)
        f_hat_2 = self.get_normalized_fitness(group_2, idx_2)
        sign = 1 if group_1 == "A" else -1
        prob = self.probability_formula(f_hat_1, f_hat_2, sign)
        return prob

    def probability_formula(self, f_hat_1: float, f_hat_2: float, sign: int):
        """Apply the probability formula"""
        return (1 + np.exp(sign * self.eta * (f_hat_2 - f_hat_1))) ** (-1)

    def update_extreme_fitness(self):
        """Update the min/max fitness"""
        for group in self.groups:
            _, fitness = self.get_current_info_group(group)
            self.extreme_fitness[group]["min"].append(np.min(fitness))
            self.extreme_fitness[group]["max"].append(np.max(fitness))

    # --------------------------#
    # --------------------------#
    # ----    ACCESS INFO   ----#
    # --------------------------#
    # --------------------------#

    def get_normalized_fitness(self, group, idx):
        """Compute normalize fitness of group[idx] agent."""
        fitness = self.population[group][idx].fitness
        min_fit, max_fit = (
            self.extreme_fitness[group]["min"][-1],
            self.extreme_fitness[group]["max"][-1],
        )
        eps = 1e-10
        return (fitness - min_fit + eps) / (max_fit - min_fit + eps)

    def get_current_info_group(self, group: str):
        positions = []
        fitness = []
        for idx_agent in self.population[group]:
            positions.append(self.population[group][idx_agent].position)
            fitness.append(self.population[group][idx_agent].fitness)

        return np.array(positions), np.array(fitness)

    # --------------------------#
    # --------------------------#
    # ----  INITIALIZATION  ----#
    # --------------------------#
    # --------------------------#

    def init_population(self):
        """Init the population with random positions"""
        pop = {group: {} for group in self.groups}
        for group, n_group in self.demography.items():
            for idx in range(n_group):
                init_pos = self.get_random_position()
                pop[group][idx] = Agent(group, init_pos, self.init_fitness, idx)

        return pop

    def init_extreme_fitness(self):
        """Init the extreme fitness."""
        init_fit = self.init_fitness
        extreme_fitness = {
            group: {"min": [init_fit], "max": [init_fit]}
            for group in self.groups
        }
        return extreme_fitness

    def get_random_position(self):
        """Chose a random position on the grid"""
        random_x = np.random.randint(0, self.grid_size)
        random_y = np.random.randint(0, self.grid_size)
        return (random_x, random_y)

    # --------------------------#
    # --------------------------#
    # ----  PLOTTING UTILS  ----#
    # --------------------------#
    # --------------------------#

    def plot_fitness(self, t_max, plot: bool = False):
        """Plot the fitness evolution during the game."""
        for group in self.groups:
            for idx in tqdm.tqdm(self.population[group]):
                self.population[group][idx].position_hist = 0

        for group in self.groups:
            plt.figure()
            plt.title(f"Group {group}")
            for idx in tqdm.tqdm(self.population[group]):
                assert (
                    len(self.population[group][idx].fitness_hist) == t_max + 1
                )
                plt.plot(self.population[group][idx].fitness_hist)
                plt.grid()
            plt.savefig(os.path.join(self.path, f"fitness_{group}.png"))
            if plot:
                plt.show()
            plt.close()

    def plot_current_situation(self, step: int, plot: bool):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Time {step}")
        for idx, group in enumerate(self.groups):
            position_b, fitness_b = self.get_current_info_group(group)
            sc2 = axes[idx].scatter(
                position_b[:, 0],
                position_b[:, 1],
                c=fitness_b,
                cmap=self.plot_colors[group],
                vmin=np.min(fitness_b),
                vmax=np.max(fitness_b),
            )
            fig.colorbar(sc2, ax=axes[idx], label="Fitness")
            axes[idx].set_title(f"Group {group}")
            axes[idx].grid()
            axes[idx].set_xticks(np.arange(self.grid_size + 1))
            axes[idx].set_yticks(np.arange(self.grid_size + 1))
            axes[idx].set_aspect("equal")
            axes[idx].set_xlim((-0.5, self.grid_size + 0.5))
            axes[idx].set_ylim((-0.5, self.grid_size + 0.5))

        plt.savefig(os.path.join(self.path_grid, f"{str(step)}.png"), dpi=100)
        if plot:
            plt.show()
        plt.close()
