import numpy as np
import tqdm
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.bonabeau.agent_bonabeau import Agent


"""game class"""

import numpy as np
import tqdm


class Game:
    def __init__(
        self,
        N: int,
        mu: float,
        grid_size: int = 20,
        eta: float = 10,
        tracked_fixed: bool = True,
        plot_final: bool = True,
    ):
        # input args
        self.N = N
        self.mu = mu
        self.grid_size = grid_size
        self.eta = eta
        self.plot_final = plot_final

        # deduced args
        self.rho = self.N / (self.grid_size**2)

        # population and grid (fitness init is included)
        self.population = []
        self.grid = [[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.init_population()

        # plots
        if tracked_fixed:
            self.tracked_ids = list(range(10))
        else:
            self.tracked_ids = np.random.choice(self.N, size=10, replace=False)

    def play_game(self, t_max: int, Pi_update_every: int = 1):
        print(f"Rho : {self.rho}")
        for t in tqdm.tqdm(range(t_max)):
            self.play_one_iter(update_Pi=(t % Pi_update_every == 0))
        if self.plot_final:
            self.plot_tracked_Pi()

    def play_one_iter(self, update_Pi: bool = True):
        # move
        self.move_all_agents()

        # fight
        self.check_and_do_fight()

        # <Pi>
        for agent in self.population:
            relaxed_fitness = agent.fitness - self.mu * np.tanh(agent.fitness)
            agent.fitness_hist[-1] = relaxed_fitness

        if update_Pi:
            self.update_all_Pi()
        else:
            for agent in self.population:
                agent.Pi_hist.append(agent.Pi)

        for agent in self.population:
            D, S = agent.fights_won, agent.fights_lost
            if D + S > 0:
                agent.Xi = D / (D + S)
            else:
                agent.Xi = 0.0

    def check_and_do_fight(self):
        """Checks if a fight should take place and does it in such a case."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                agents = self.grid[x][y]
                if agents and len(agents) > 1:
                    self.fight(agents[0], agents[1])
                    for a in agents[2:]:
                        a.add_unchanged_step()
                elif agents and len(agents) == 1:
                    agents[0].add_unchanged_step()

    def move_all_agents(self):
        """Move all agents by one step and update the grid."""
        # reset grid
        self.grid = [[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for agent in self.population:
            agent.move_position(self.grid_size)
            x, y = agent.position
            self.grid[x][y].append(agent)

    def fight(self, agent_i, agent_j):
        hi, hj = agent_i.fitness, agent_j.fitness
        prob_i_wins = 1 / (1 + np.exp(self.eta * (hj - hi)))

        if np.random.rand() < prob_i_wins:
            agent_i.fights_won += 1
            agent_j.fights_lost += 1
            agent_i.fitness_hist.append(hi + 1)
            agent_j.fitness_hist.append(hj - 1)
        else:
            agent_j.fights_won += 1
            agent_i.fights_lost += 1
            agent_i.fitness_hist.append(hi - 1)
            agent_j.fitness_hist.append(hj + 1)

    def update_all_Pi(self):
        """Vectorized version: compute Pi for all agents using numpy."""
        fitnesses = np.array([agent.fitness for agent in self.population])

        diff_matrix = fitnesses[np.newaxis, :] - fitnesses[:, np.newaxis]
        np.fill_diagonal(diff_matrix, np.nan)

        P = 1 / (1 + np.exp(self.eta * diff_matrix))
        Pi_vector = np.nanmean(P, axis=1)
        for agent, Pi in zip(self.population, Pi_vector):
            agent.Pi = Pi

    def compute_sigma2(self):
        """Compute the global hierarchy measure sigma2² = sum (Xi - 0.5)^2"""
        Xi_values = np.array([agent.Xi for agent in self.population])
        sigma2 = np.sum((Xi_values - 0.5) ** 2)
        return sigma2

    # _____________________________________ INIT _____________________________________

    def init_population(self):
        """Place N agents at random positions on the grid with initial fitness = 0.0"""
        positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
        ]

        if self.N > len(positions):
            raise ValueError("Grid is too small to place all agents without overlap.")

        np.random.shuffle(positions)

        for i in range(self.N):
            pos = positions[i]
            agent = Agent(init_position=pos, init_fitness=0.0, id=i)
            self.population.append(agent)
            x, y = pos
            self.grid[x][y].append(agent)

    # _____________________________________ PLOTS _____________________________________

    def plot_tracked_Pi(self):
        plt.figure(figsize=(6, 4))
        for i in self.tracked_ids:
            agent = self.population[i]
            plt.plot(agent.Pi_hist, label=f"Agent {i}")
        plt.xlabel("Iterations")
        plt.ylabel("$<P_i>$")
        plt.title(
            "Evolution of the instantaneous probability that individual $i$ wins in a random fight"
        )
        plt.legend(loc="center right")
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.show()

    def animate_agents(
        self, interval=100, save_gif=False, gif_name="animation.gif", n_frames=100
    ):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1, self.grid_size)
        ax.set_ylim(-1, self.grid_size)
        ax.set_title("")

        scatter = ax.scatter([], [], c=[], cmap="viridis", s=80, edgecolors="k")
        norm = mcolors.Normalize(vmin=0, vmax=1)

        # Ajouter la colorbar
        cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
        cbar.set_label(r"$\Pi$ (hiérarchie sociale)")

        def init():
            scatter.set_offsets(np.empty((0, 2)))
            scatter.set_array(np.array([]))
            return (scatter,)

        def update(frame):
            self.play_one_iter()

            # Positions et Pi
            positions = np.vstack([agent.position for agent in self.population])
            Pi_values = np.array([agent.Pi for agent in self.population])

            # Mettre à jour le scatter plot
            scatter.set_offsets(positions)
            scatter.set_array(Pi_values)
            scatter.set_norm(norm)

            # Mettre à jour le titre avec le numéro de frame
            ax.set_title(f"Évolution des agents — itération {frame + 1}", fontsize=12)

            return (scatter,)

        anim = FuncAnimation(
            fig, update, init_func=init, frames=n_frames, interval=interval, blit=True
        )

        if save_gif:
            anim.save(gif_name, writer="pillow", fps=1000 // interval)
        else:
            plt.show()
