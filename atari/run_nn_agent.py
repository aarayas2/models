import torch
import pygame
from policy_network import GridPolicyNetwork
from nn_grid_atari import GridPong, GridBreakout, GridPacman, GridFreeway

# Map selection to environment classes
games = {
    "1": ("Pong", GridPong),
    "2": ("Breakout", GridBreakout),
    "3": ("Pacman", GridPacman),
    "4": ("Freeway", GridFreeway),
}

print("Select Game for Untrained NN to play:")
print("1: Pong\n2: Breakout\n3: Pacman\n4: Freeway")
choice = input("Choice (1-4) [default: 1]: ").strip() or "1"
game_name, env_cls = games.get(choice, games["1"])

# 1. Initialize the Environment and Neural Network
env = env_cls(render_mode="human")
model = GridPolicyNetwork()

# Set model to evaluation mode (disables gradients/dropout if we had any)
model.eval()

obs, _ = env.reset()
env.render()

print(f"\n--- Untrained NN Playing {game_name} ---")
print("Watch the random weights try to play. Close window to exit.\n")

running = True
try:
    while running:
        # Handle window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if not running:
            break

        # 2. Convert NumPy observation (5, 10, 10) to PyTorch Tensor (1, 5, 10, 10)
        # We add the batch dimension of size 1 using unsqueeze(0)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # 3. Forward pass through the network (no gradient tracking needed for inference)
        with torch.no_grad():
            logits = model(obs_tensor)
        
        # 4. Action Selection
        # Option A (Deterministic): Always pick the action with the highest logit
        action = torch.argmax(logits, dim=-1).item()
        
        # Option B (Stochastic): Sample from a probability distribution (useful for exploration)
        # probs = torch.softmax(logits, dim=-1)
        # action = torch.multinomial(probs, 1).item()

        # 5. Step the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated or truncated:
            print(f"Game Over! Reward: {reward}. Resetting...")
            obs, _ = env.reset()
            env.render()
            
finally:
    env.close()