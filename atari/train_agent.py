import os
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from policy_network import GridPolicyNetwork
from nn_grid_atari import GridPong, GridBreakout, GridPacman, GridFreeway

# Automatically select CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

games = {
    "1": ("Pong", GridPong, "pong_weights.pt"),
    "2": ("Breakout", GridBreakout, "breakout_weights.pt"),
    "3": ("Pacman", GridPacman, "pacman_weights.pt"),
    "4": ("Freeway", GridFreeway, "freeway_weights.pt"),
}

print("\nSelect Game to Train:")
print("1: Pong\n2: Breakout\n3: Pacman\n4: Freeway")
choice = input("Choice (1-4) [default: 2]: ").strip() or "2"
game_name, env_cls, save_filename = games.get(choice, games["2"])

# Headless environment for fast step throughput
env = env_cls(render_mode=None)

# Send model to selected device (CUDA or CPU)
policy = GridPolicyNetwork().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

GAMMA = 0.99
NUM_EPISODES = 2000
MAX_STEPS = 200

print(f"\n--- Training {game_name} for {NUM_EPISODES} episodes on {device} ---")
recent_rewards = []

for episode in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset()
    log_probs = []
    rewards = []

    for step in range(MAX_STEPS):
        # Convert observation and send tensor to CUDA
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        
        logits = policy(obs_tensor)

        # Categorical distribution operates directly on device
        dist = Categorical(logits=logits)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        # Action is transferred to CPU host for NumPy environment step
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)

        if terminated or truncated:
            break

    total_episode_reward = sum(rewards)
    recent_rewards.append(total_episode_reward)
    if len(recent_rewards) > 50:
        recent_rewards.pop(0)

    # Compute discounted returns
    discounted_returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + GAMMA * R
        discounted_returns.insert(0, R)

    returns = torch.tensor(discounted_returns, dtype=torch.float32, device=device)

    # Normalize returns on device for baseline stability
    if len(returns) > 1 and returns.std() > 1e-7:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Calculate policy loss directly on device
    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()

    if episode % 50 == 0:
        avg_reward = np.mean(recent_rewards)
        print(f"Episode {episode:4d} | Avg Reward (last 50): {avg_reward:6.2f}")

env.close()

# Save state_dict transferred to CPU so it can be loaded on any hardware
torch.save(policy.to("cpu").state_dict(), save_filename)
print(f"\nTraining complete! Network saved to: {os.path.abspath(save_filename)}")