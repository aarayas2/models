import time
import ale_py
import gymnasium as gym

# Select one of: "ALE/Pong-v5", "ALE/Breakout-v5", "ALE/MsPacman-v5", "ALE/Freeway-v5"
GAME_ID = "ALE/Pong-v5"

# Initialize the environment with human visual rendering
env = gym.make(GAME_ID, render_mode="human")

# Reset the environment to its initial state
observation, info = env.reset()

print(f"Loaded: {GAME_ID}")
print(f"Action space: {env.action_space}")
print(f"Observation shape: {observation.shape}")

# Run 1,000 steps taking random actions
for step in range(1000):
    # Sample a random valid action from the action space
    action = env.action_space.sample()

    # Step the environment forward
    observation, reward, terminated, truncated, info = env.step(action)

    # Slow down loop slightly to match normal viewing speed (~60 FPS)
    time.sleep(1 / 60)

    # Reset if a game round ends or reaches truncation limit
    if terminated or truncated:
        observation, info = env.reset()

env.close()