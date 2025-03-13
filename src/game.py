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
        # input args
        self.n_a = n_a
        self.n_b = n_b
        self.grid_size = grid_size
        self.eta = eta
        self.x = x

        # deduced args
        self.n = n_a + n_b
        self.init_fitness = 1000 / self.n

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

    def play_game(self, t_max: int):
        print(len(self.population["A"]), len(self.population["B"]))
        for t in tqdm.tqdm(range(t_max)):
            # print(t)
            self.play_one_iter()
            self.plot_current_situation(t)

        create_gif_from_images(self.path_grid, self.path_grid)
        self.plot_fitness(t_max)

    def plot_fitness(self, t_max):
        plt.figure()
        group = "A"
        for idx in tqdm.tqdm(self.population[group]):
            assert len(self.population[group][idx].fitness_hist) == t_max + 1
            plt.plot(self.population[group][idx].fitness_hist)
        plt.show()

        plt.figure()
        group = "B"
        for idx in tqdm.tqdm(self.population[group]):
            assert len(self.population[group][idx].fitness_hist) == t_max + 1
            plt.plot(self.population[group][idx].fitness_hist)
        plt.show()

    def play_one_iter(self):
        # move
        moving_group, moving_agent = self.move_agent()
        new_pos = np.array(self.population[moving_group][moving_agent].position)

        # checking for fight
        unmoving_group = "A" if moving_group == "B" else "B"
        position_unmov, _ = self.get_current_info_group(unmoving_group)
        fight_locations = np.argwhere(
            np.abs(new_pos - position_unmov).sum(axis=1) < 1e-4
        )
        if fight_locations.size > 0:
            # print("Fight")
            unmoving_agent = np.random.choice(fight_locations[0])
            self.fight(
                moving_group, moving_agent, unmoving_group, unmoving_agent
            )
            # print(
            #     unmoving_group,
            #     unmoving_agent,
            #     np.abs(new_pos - position_unmov).sum(axis=1).shape,
            # )
        else:
            fit = self.population[moving_group][moving_agent].fitness
            self.population[moving_group][moving_agent].fitness_hist.append(fit)
            unmoving_agent = None

        self.update_non_active_agents(
            moving_group, moving_agent, unmoving_group, unmoving_agent
        )

    def update_non_active_agents(self, group_1, idx_1, group_2, idx_2):
        for idx_agent_1 in self.population[group_1]:
            if idx_agent_1 != idx_1:
                self.population[group_1][idx_agent_1].add_unchanged_step()
        for idx_agent_2 in self.population[group_2]:
            if idx_agent_2 != idx_2:
                self.population[group_2][idx_agent_2].add_unchanged_step()

    def move_agent(self):
        moving_group = np.random.choice(
            ["A", "B"], p=[self.n_a / self.n, self.n_b / self.n]
        )
        moving_agent = np.random.choice(
            list(self.population[moving_group].keys())
        )
        self.population[moving_group][moving_agent].move_position(
            self.grid_size
        )

        return moving_group, moving_agent

    def fight(self, group_1, idx_1, group_2, idx_2):
        f_hat_1 = self.get_normalized_fitness(group_1, idx_1)
        f_hat_2 = self.get_normalized_fitness(group_2, idx_2)
        assert f_hat_1 > 0 and f_hat_1 <= 1
        assert f_hat_2 > 0 and f_hat_2 <= 1
        sign = 1 if group_1 == "A" else -1
        prob = (1 + np.exp(sign * self.eta * (f_hat_2 - f_hat_1))) ** (-1)
        u = np.random.rand()
        fit_1 = self.population[group_1][idx_1].fitness
        fit_2 = self.population[group_2][idx_2].fitness
        if u < prob:
            if group_1 == "A":
                # print("Vic", "a")
                self.population[group_1][idx_1].fitness_hist.append(
                    fit_1 + self.x * fit_2
                )
                self.population[group_2][idx_2].fitness_hist.append(
                    fit_2 - self.x * fit_2
                )
            else:
                # print("Vic", "b")

                self.population[group_1][idx_1].fitness_hist.append(
                    fit_1 - self.x * fit_1
                )
                self.population[group_2][idx_2].fitness_hist.append(
                    fit_2 + self.x * fit_1
                )
        else:
            if group_1 == "A":
                # print("Def", "a")

                self.population[group_1][idx_1].fitness_hist.append(
                    fit_1 - self.x * fit_1
                )
                self.population[group_2][idx_2].fitness_hist.append(
                    fit_2 + self.x * fit_1
                )
            else:
                # print("Def", "b")

                self.population[group_1][idx_1].fitness_hist.append(
                    fit_1 + self.x * fit_2
                )
                self.population[group_2][idx_2].fitness_hist.append(
                    fit_2 - self.x * fit_2
                )

        self.update_extreme_fitness()

    def update_extreme_fitness(self):
        _, fitness_a = self.get_current_info_group("A")
        self.extreme_fitness["A"]["min"].append(np.min(fitness_a))
        self.extreme_fitness["A"]["max"].append(np.max(fitness_a))

        _, fitness_b = self.get_current_info_group("B")
        self.extreme_fitness["B"]["min"].append(np.min(fitness_b))
        self.extreme_fitness["B"]["max"].append(np.max(fitness_b))

    def get_normalized_fitness(self, group, idx):
        fitness = self.population[group][idx].fitness
        min_fit, max_fit = (
            self.extreme_fitness[group]["min"][-1],
            self.extreme_fitness[group]["max"][-1],
        )
        return (fitness - min_fit + 1e-5) / (max_fit - min_fit + 1e-5)

    def init_population(self):
        pop = {"A": {}, "B": {}}
        for idx_a in range(self.n_a):
            init_pos = self.get_random_position()
            pop["A"][idx_a] = Agent("A", init_pos, self.init_fitness, idx_a)

        for idx_b in range(self.n_b):
            init_pos = self.get_random_position()
            pop["B"][idx_b] = Agent("B", init_pos, self.init_fitness, idx_b)

        return pop

    def init_extreme_fitness(self):
        init_fit = self.init_fitness
        extreme_fitness = {
            "A": {"min": [init_fit], "max": [init_fit]},
            "B": {"min": [init_fit], "max": [init_fit]},
        }
        return extreme_fitness

    def get_random_position(self):
        random_x = np.random.randint(0, self.grid_size)
        random_y = np.random.randint(0, self.grid_size)
        return (random_x, random_y)

    def get_current_info_group(self, group: str):
        positions = []
        fitness = []
        for idx_agent in self.population[group]:
            positions.append(self.population[group][idx_agent].position)
            fitness.append(self.population[group][idx_agent].fitness)

        return np.array(positions), np.array(fitness)

    def plot_current_situation(self, step: int):
        # can be easily adapted with arbitrary number of groups
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        fig.suptitle(f"Time {step}")
        # group A
        position_a, fitness_a = self.get_current_info_group("A")
        sc1 = axes[0].scatter(
            position_a[:, 0],
            position_a[:, 1],
            c=fitness_a,
            cmap=self.plot_colors["A"],
            vmin=1.8,
            vmax=1.9,
        )
        fig.colorbar(sc1, ax=axes[0], label="Fitness")
        axes[0].set_title("Group A")
        axes[0].grid()
        axes[0].set_xticks(np.arange(self.grid_size))
        axes[0].set_yticks(np.arange(self.grid_size))
        axes[0].set_aspect("equal")
        axes[0].set_xlim((0, self.grid_size))
        axes[0].set_ylim((0, self.grid_size))

        # group B
        position_b, fitness_b = self.get_current_info_group("B")
        sc2 = axes[1].scatter(
            position_b[:, 0],
            position_b[:, 1],
            c=fitness_b,
            cmap=self.plot_colors["B"],
            vmin=1.8,
            vmax=1.9,
        )
        fig.colorbar(sc2, ax=axes[1], label="Fitness")
        axes[1].set_title("Group B")
        axes[1].grid()
        axes[1].set_xticks(np.arange(self.grid_size))
        axes[1].set_yticks(np.arange(self.grid_size))
        axes[1].set_aspect("equal")
        axes[1].set_xlim((0, self.grid_size))
        axes[1].set_ylim((0, self.grid_size))

        plt.savefig(os.path.join(self.path_grid, f"{str(step)}.png"), dpi=100)
        plt.close()
