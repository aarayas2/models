import os
import torch
import pygame
from policy_network import GridPolicyNetwork
from nn_grid_atari import GridPong, GridBreakout, GridPacman, GridFreeway

games = {
    "1": ("Pong", GridPong, "pong_weights.pt"),
    "2": ("Breakout", GridBreakout, "breakout_weights.pt"),
    "3": ("Pacman", GridPacman, "pacman_weights.pt"),
    "4": ("Freeway", GridFreeway, "freeway_weights.pt"),
}

print("Select Game to Watch:")
print("1: Pong\n2: Breakout\n3: Pacman\n4: Freeway")
choice = input("Choice (1-4) [default: 2]: ").strip() or "2"
game_name, env_cls, weight_file = games.get(choice, games["2"])

env = env_cls(render_mode="human")
model = GridPolicyNetwork()

# Check and load trained weights if they exist
if os.path.exists(weight_file):
    model.load_state_dict(torch.load(weight_file))
    print(f"Loaded trained weights from: {weight_file}")
else:
    print(f"No saved weights found at '{weight_file}'. Running with random weights.")

model.eval()
obs, _ = env.reset()
env.render()

print(f"\n--- Running {game_name} ---")
running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not running:
            break

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(obs_tensor)

        # Greedily pick the action with highest probability
        action = torch.argmax(logits, dim=-1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated or truncated:
            obs, _ = env.reset()
            env.render()
finally:
    env.close()