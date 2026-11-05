import os
import torch
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from policy_network import GridPolicyNetwork
from nn_grid_atari import GridPong, GridBreakout, GridPacman, GridFreeway

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

env = env_cls(render_mode=None)

# Initialize network and move to device
policy = GridPolicyNetwork().to(device)

# --- CHECK FOR EXISTING CHECKPOINT ---
if os.path.exists(save_filename):
    print(f"\n[+] Found existing checkpoint: '{save_filename}'. Resuming training from saved weights.")
    state_dict = torch.load(save_filename, map_location=device)
    policy.load_state_dict(state_dict)
else:
    print(f"\n[-] No existing checkpoint found for '{save_filename}'. Starting training from scratch.")

optimizer = optim.Adam(policy.parameters(), lr=1e-3)

GAMMA = 0.99
NUM_EPISODES = 2000
MAX_STEPS = 200

print(f"\n--- Training {game_name} for {NUM_EPISODES} additional episodes on {device} ---")
recent_rewards = []

for episode in range(1, NUM_EPISODES + 1):
    obs, _ = env.reset()
    log_probs = []
    rewards = []

    for step in range(MAX_STEPS):
        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        
        logits = policy(obs_tensor)

        dist = Categorical(logits=logits)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

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

    # Normalize returns
    if len(returns) > 1 and returns.std() > 1e-7:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

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

# Save updated weights (ensuring CPU portability)
torch.save(policy.to("cpu").state_dict(), save_filename)
print(f"\nTraining session complete! Updated weights saved to: {os.path.abspath(save_filename)}")