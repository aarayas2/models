import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

from policy_network import GridPolicyNetwork
from nn_grid_atari import GridPong, GridBreakout, GridPacman, GridFreeway

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Critic network used strictly during training to reduce variance
class GridValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(500, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

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

# Target policy matching the exact 17,253 weight spec
actor = GridPolicyNetwork().to(device)
critic = GridValueNetwork().to(device)

actor_opt = optim.Adam(actor.parameters(), lr=3e-4)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

# PPO Hyperparameters
STEPS_PER_BATCH = 2048
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.02   # Prevents action collapse to the wall
TOTAL_TIMESTEPS = 150_000

obs, _ = env.reset()
global_step = 0
iteration = 0

print(f"\n--- Training {game_name} via PPO for {TOTAL_TIMESTEPS:,} total steps ---")

while global_step < TOTAL_TIMESTEPS:
    b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values = [], [], [], [], [], []

    # 1. Collect rollout data
    for _ in range(STEPS_PER_BATCH):
        global_step += 1
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = actor(obs_tensor)
            val = critic(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        b_obs.append(obs)
        b_actions.append(action.item())
        b_logprobs.append(logprob.item())
        b_rewards.append(reward)
        b_dones.append(done)
        b_values.append(val.item())

        obs = next_obs
        if done:
            obs, _ = env.reset()

    # 2. Compute Generalized Advantage Estimation (GAE)
    with torch.no_grad():
        next_val = critic(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()
    
    advantages = np.zeros(STEPS_PER_BATCH, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(STEPS_PER_BATCH)):
        next_non_terminal = 1.0 - b_dones[t]
        next_v = next_val if t == STEPS_PER_BATCH - 1 else b_values[t + 1]
        delta = b_rewards[t] + GAMMA * next_v * next_non_terminal - b_values[t]
        advantages[t] = last_gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae

    returns = advantages + np.array(b_values)

    # Convert rollout to torch tensors
    t_obs = torch.tensor(np.array(b_obs), dtype=torch.float32, device=device)
    t_actions = torch.tensor(np.array(b_actions), dtype=torch.long, device=device)
    t_old_logprobs = torch.tensor(np.array(b_logprobs), dtype=torch.float32, device=device)
    t_adv = torch.tensor(advantages, dtype=torch.float32, device=device)
    t_returns = torch.tensor(returns, dtype=torch.float32, device=device)
    t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)

    # 3. Optimize surrogate policy and value objective
    indices = np.arange(STEPS_PER_BATCH)
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, STEPS_PER_BATCH, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_idx = indices[start:end]

            # Forward pass
            logits = actor(t_obs[mb_idx])
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(t_actions[mb_idx])
            entropy = dist.entropy().mean()

            # Policy loss with PPO clipping
            ratio = torch.exp(new_logprobs - t_old_logprobs[mb_idx])
            surr1 = ratio * t_adv[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * t_adv[mb_idx]
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEFF * entropy

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # Value loss
            new_values = critic(t_obs[mb_idx]).squeeze(-1)
            critic_loss = 0.5 * ((new_values - t_returns[mb_idx]) ** 2).mean()

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

    iteration += 1
    recent_mean_reward = np.sum(b_rewards) / max(1, sum(b_dones))
    print(f"Iter {iteration:2d} | Timesteps: {global_step:7d} | Avg Ep Reward: {recent_mean_reward:6.2f}")

env.close()

# Save only the actor's weights (17,253 params)
torch.save(actor.to("cpu").state_dict(), save_filename)
print(f"\nTraining complete! Actor saved to: {os.path.abspath(save_filename)}")