import time
import os
from grid_atari import GridBreakout, GridPong, GridPacman, GridFreeway

games = {
    "1": ("Breakout", GridBreakout()),
    "2": ("Pong", GridPong()),
    "3": ("Pacman", GridPacman()),
    "4": ("Freeway", GridFreeway()),
}

print("Select Game to Test:")
print("1: Breakout\n2: Pong\n3: Pacman\n4: Freeway")
choice = input("Choice (1-4) [default: 1]: ").strip() or "1"
game_name, env = games.get(choice, games["1"])

obs, _ = env.reset()

print(f"\nRunning {game_name} (Observation shape: {obs.shape}, Actions: 0-4)")

for step in range(60):
    # Select random action from {0: NOOP, 1: UP, 2: DOWN, 3: LEFT, 4: RIGHT}
    action = env.action_space.sample()
    obs, reward, terminated, truncated, _ = env.step(action)

    # Clear terminal screen
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"Game: {game_name} | Step: {step} | Action: {action} | Reward: {reward}")
    print(env.render())
    time.sleep(0.15)

    if terminated or truncated:
        obs, _ = env.reset()