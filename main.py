from src.game import Game

if __name__ == "__main__":
    game_params = {"n_a": 500, "n_b": 50, "x": 0.05, "eta": 5, "grid_size": 25}
    game = Game(**game_params)
    game.play_game(t_max=200_001, save_every=10_000)
