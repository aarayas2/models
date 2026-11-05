import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

# Shared 5 actions
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

class BaseGridEnv(gym.Env):
    """Base class handling Pygame window management and crisp pixel scaling."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, render_mode=None, cell_size=40):
        super().__init__()
        self.grid_size = 10
        self.cell_size = cell_size
        self.window_size = self.grid_size * cell_size  # 400x400
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(10, 10, 3), dtype=np.uint8
        )

        # FIX: Initialize Pygame immediately so the window exists for event checks
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption(self.__class__.__name__)
            self.clock = pygame.time.Clock()

    def _render_frame(self):
        rgb_grid = self._get_color_grid()

        if self.render_mode == "human":
            # Scale 10x10 up to window_size using nearest-neighbor
            surface = pygame.surfarray.make_surface(np.transpose(rgb_grid, (1, 0, 2)))
            scaled_surface = pygame.transform.scale(surface, (self.window_size, self.window_size))

            self.window.blit(scaled_surface, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        return rgb_grid

    def render(self):
        if self.render_mode in ["human", "rgb_array"]:
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


# -------------------------------------------------------------
# 1. PONG
# -------------------------------------------------------------
class GridPong(BaseGridEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_y = 4
        self.opp_y = 4
        self.ball = [5, 5]
        self.ball_vel = [1, 1]
        return self._get_color_grid(), {}

    def _get_color_grid(self):
        grid = np.zeros((10, 10, 3), dtype=np.uint8)
        grid[:, :] = [34, 139, 34]  # Green court
        grid[:, 4:6] = [40, 160, 40]
        grid[self.opp_y : self.opp_y + 2, 0] = [220, 60, 50]
        grid[self.player_y : self.player_y + 2, 9] = [255, 255, 255]
        grid[self.ball[0], self.ball[1]] = [255, 255, 100]
        return grid

    def step(self, action):
        if action == UP:
            self.player_y = max(0, self.player_y - 1)
        elif action == DOWN:
            self.player_y = min(8, self.player_y + 1)

        if self.ball[0] > self.opp_y and self.opp_y < 8:
            self.opp_y += 1
        elif self.ball[0] < self.opp_y and self.opp_y > 0:
            self.opp_y -= 1

        self.ball[0] += self.ball_vel[0]
        self.ball[1] += self.ball_vel[1]

        if self.ball[0] <= 0:
            self.ball[0] = 0
            self.ball_vel[0] *= -1
        elif self.ball[0] >= 9:
            self.ball[0] = 9
            self.ball_vel[0] *= -1

        reward, terminated = 0.0, False

        if self.ball[1] <= 0:
            if self.opp_y <= self.ball[0] <= self.opp_y + 1:
                self.ball_vel[1] = 1
            else:
                reward, terminated = 1.0, True

        if self.ball[1] >= 9:
            if self.player_y <= self.ball[0] <= self.player_y + 1:
                self.ball_vel[1] = -1
                reward = 0.2
            else:
                reward, terminated = -1.0, True

        return self._get_color_grid(), reward, terminated, False, {}


# -------------------------------------------------------------
# 2. BREAKOUT
# -------------------------------------------------------------
class GridBreakout(BaseGridEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.paddle_x = 4
        self.ball = [7, 5]
        self.ball_vel = [-1, 1]
        self.bricks = np.zeros((10, 10), dtype=bool)
        self.bricks[1:3, 1:9] = True
        return self._get_color_grid(), {}

    def _get_color_grid(self):
        grid = np.zeros((10, 10, 3), dtype=np.uint8)
        grid[:, :] = [18, 18, 24]
        for r in range(1, 3):
            color = [220, 50, 40] if r == 1 else [240, 150, 30]
            for c in range(10):
                if self.bricks[r, c]:
                    grid[r, c] = color
        grid[9, self.paddle_x : self.paddle_x + 2] = [40, 180, 220]
        grid[self.ball[0], self.ball[1]] = [255, 255, 255]
        return grid

    def step(self, action):
        if action == LEFT:
            self.paddle_x = max(0, self.paddle_x - 1)
        elif action == RIGHT:
            self.paddle_x = min(8, self.paddle_x + 1)

        self.ball[0] += self.ball_vel[0]
        self.ball[1] += self.ball_vel[1]

        if self.ball[1] <= 0:
            self.ball[1] = 0
            self.ball_vel[1] *= -1
        elif self.ball[1] >= 9:
            self.ball[1] = 9
            self.ball_vel[1] *= -1

        if self.ball[0] <= 0:
            self.ball[0] = 0
            self.ball_vel[0] *= -1

        reward, terminated = 0.0, False

        if self.bricks[self.ball[0], self.ball[1]]:
            self.bricks[self.ball[0], self.ball[1]] = False
            self.ball_vel[0] *= -1
            reward = 1.0

        if self.ball[0] == 9:
            if self.paddle_x <= self.ball[1] <= self.paddle_x + 1:
                self.ball_vel[0] = -1
                reward += 0.2
            else:
                terminated = True

        if not self.bricks.any():
            reward += 5.0
            terminated = True

        return self._get_color_grid(), reward, terminated, False, {}


# -------------------------------------------------------------
# 3. PAC-MAN
# -------------------------------------------------------------
class GridPacman(BaseGridEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pacman = [8, 5]
        self.ghost = [1, 5]
        self.dots = np.zeros((10, 10), dtype=bool)
        self.dots[2:8, 2:8] = True
        return self._get_color_grid(), {}

    def _get_color_grid(self):
        grid = np.zeros((10, 10, 3), dtype=np.uint8)
        grid[:, :] = [10, 10, 35]
        for r in range(10):
            for c in range(10):
                if self.dots[r, c]:
                    grid[r, c] = [240, 200, 100]
        grid[self.ghost[0], self.ghost[1]] = [240, 40, 40]
        grid[self.pacman[0], self.pacman[1]] = [255, 230, 0]
        return grid

    def step(self, action):
        if action == UP:
            self.pacman[0] = max(0, self.pacman[0] - 1)
        elif action == DOWN:
            self.pacman[0] = min(9, self.pacman[0] + 1)
        elif action == LEFT:
            self.pacman[1] = max(0, self.pacman[1] - 1)
        elif action == RIGHT:
            self.pacman[1] = min(9, self.pacman[1] + 1)

        if self.ghost[0] < self.pacman[0]:
            self.ghost[0] += 1
        elif self.ghost[0] > self.pacman[0]:
            self.ghost[0] -= 1
        elif self.ghost[1] < self.pacman[1]:
            self.ghost[1] += 1
        elif self.ghost[1] > self.pacman[1]:
            self.ghost[1] -= 1

        reward, terminated = 0.0, False

        if self.dots[self.pacman[0], self.pacman[1]]:
            self.dots[self.pacman[0], self.pacman[1]] = False
            reward = 1.0

        if self.pacman == self.ghost:
            reward = -5.0
            terminated = True
        elif not self.dots.any():
            reward += 10.0
            terminated = True

        return self._get_color_grid(), reward, terminated, False, {}


# -------------------------------------------------------------
# 4. FREEWAY
# -------------------------------------------------------------
class GridFreeway(BaseGridEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.chicken = [9, 5]
        self.car_lanes = [
            {"row": r, "col": float(np.random.randint(0, 10)), "speed": 0.5 * (1 if r % 2 == 0 else -1)}
            for r in range(1, 9)
        ]
        return self._get_color_grid(), {}

    def _get_color_grid(self):
        grid = np.zeros((10, 10, 3), dtype=np.uint8)
        grid[1:9, :] = [45, 45, 50]
        grid[0, :] = [34, 139, 34]
        grid[9, :] = [34, 139, 34]
        for car in self.car_lanes:
            c_idx = int(car["col"]) % 10
            color = [230, 200, 30] if car["speed"] > 0 else [220, 60, 60]
            grid[car["row"], c_idx] = color
        grid[self.chicken[0], self.chicken[1]] = [255, 255, 220]
        return grid

    def step(self, action):
        if action == UP:
            self.chicken[0] = max(0, self.chicken[0] - 1)
        elif action == DOWN:
            self.chicken[0] = min(9, self.chicken[0] + 1)
        elif action == LEFT:
            self.chicken[1] = max(0, self.chicken[1] - 1)
        elif action == RIGHT:
            self.chicken[1] = min(9, self.chicken[1] + 1)

        for car in self.car_lanes:
            car["col"] = (car["col"] + car["speed"]) % 10

        reward, terminated = 0.0, False

        for car in self.car_lanes:
            if car["row"] == self.chicken[0] and int(car["col"]) % 10 == self.chicken[1]:
                self.chicken = [9, 5]
                reward = -1.0

        if self.chicken[0] == 0:
            reward = 10.0
            terminated = True

        return self._get_color_grid(), reward, terminated, False, {}