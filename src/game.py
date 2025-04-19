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
        grid_size: int = 25,
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
        self.path = os.path.join("interactions", get_time_string())
        self.path_grid = os.path.join(self.path, "grid")
        os.mkdir(self.path)
        os.mkdir(self.path_grid)

    def play_game(self, t_max: int, save_every: int | None = None, plot: bool = False):
        """Plays according to the diagram of the paper."""
        for t in tqdm.tqdm(range(t_max)):
            self.play_one_iter()
            # plot
            if save_every is not None and t % save_every == 0:
                self.plot_current_situation(t, plot=plot)
        # plot
        self.plot_fitness(t_max, plot=plot)

    def play_one_iter(self):
        """Play one iteration of the game"""
        self.update_extreme_fitness()
        # move
        self.move_all_agents()
        self.update_all_normalized_fitness()
        self.check_and_do_fight()

    def move_all_agents(self):
        """Moves all agents (from both groups) and rebuilds the grid."""
        self.grid = [[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for group in self.groups:
            for agent in self.population[group].values():
                agent.move_position(self.grid_size)
                x, y = agent.position
                self.grid[x][y].append(agent)

    def check_and_do_fight(self):
        """For each cell, lets at most one pair of agents from opposite groups fight."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                agents = self.grid[x][y]

                if len(agents) >= 2:
                    # separe groups
                    group_A_agents = [a for a in agents if a.group == "A"]
                    group_B_agents = [a for a in agents if a.group == "B"]

                    if group_A_agents and group_B_agents:
                        # choose an agent to fight from each group
                        agent_A = np.random.choice(group_A_agents)
                        agent_B = np.random.choice(group_B_agents)

                        self.fight(agent_A, agent_B)

                        # every other agent remains spectator
                        for agent in agents:
                            if agent not in [agent_A, agent_B]:
                                agent.fitness_hist.append(agent.fitness)

                    else:  # if they are all in the same group
                        for agent in agents:
                            agent.fitness_hist.append(agent.fitness)

                elif len(agents) == 1:
                    agents[0].fitness_hist.append(agents[0].fitness)

    def fight(self, agent_1, agent_2):
        """Compute the winning probability and exchange fitness, using precomputed normalized values."""
        # Convention : agent_1 est dans le groupe A
        if agent_1.group == "B":
            agent_1, agent_2 = agent_2, agent_1

        f1, f2 = agent_1.fitness, agent_2.fitness
        f1_norm, f2_norm = agent_1.fitness_norm, agent_2.fitness_norm

        p_win = 1 / (1 + np.exp(self.eta * (f2_norm - f1_norm)))
        u = np.random.rand()

        if u < p_win:
            agent_1.fitness_hist.append(f1 + self.x * f2)
            agent_2.fitness_hist.append(f2 - self.x * f2)
        else:
            agent_1.fitness_hist.append(f1 - self.x * f1)
            agent_2.fitness_hist.append(f2 + self.x * f1)

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

    def update_all_normalized_fitness(self):
        """Compute and store normalized fitness for all agents in both groups."""
        for group in self.groups:
            f_min = self.extreme_fitness[group]["min"][-1]
            f_max = self.extreme_fitness[group]["max"][-1]
            eps = 1e-10

            for agent in self.population[group].values():
                f = agent.fitness
                agent.fitness_norm = (f - f_min + eps) / (f_max - f_min + eps)

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
            group: {"min": [init_fit], "max": [init_fit]} for group in self.groups
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
                assert len(self.population[group][idx].fitness_hist) == t_max + 1
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
