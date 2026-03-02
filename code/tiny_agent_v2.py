# tiny_agent_v2.py
# Minimal gridworld with a Q-learning agent that actually moves like it means it.

import sys
import random
import numpy as np
import pygame

# ----------------------------
# Config
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GRID = 14          # smaller world makes learning less hopeless
CELL = 28          # pixel size per grid cell
W = H = GRID * CELL
FPS_RENDER = 30

# Q-learning hyperparameters
ALPHA = 0.25       # learning rate
GAMMA = 0.95       # discount factor
EPS_START = 1.0    # epsilon-greedy start
EPS_END = 0.10     # keep a little exploration to avoid cycling
EPS_DECAY = 0.995  # decay per episode

EPISODES_TRAIN = 3000
STEPS_PER_EP = 200

# Rewards
REWARD_STEP = -0.01
REWARD_FOOD = +1.0
REWARD_POISON = -1.0
REWARD_BACKTRACK = -0.02    # penalty for immediate undo (A->B->A)
REWARD_HIT_WALL = -0.02     # penalty for trying to move into wall (no motion)

# Shaping coefficients (small; they should not drown terminal rewards)
SHAPE_TO_FOOD = 0.01        # positive if closer to food
SHAPE_FROM_POISON = 0.005   # positive if farther from poison

# Actions: down, up, right, left
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
N_ACTIONS = len(ACTIONS)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def random_empty(exclude):
    """Return a random (x, y) not in exclude."""
    while True:
        p = (random.randrange(GRID), random.randrange(GRID))
        if p not in exclude:
            return p


def greedy_action(qrow: np.ndarray) -> int:
    """Argmax with random tie-breaking."""
    mx = np.max(qrow)
    ties = np.flatnonzero(qrow == mx)
    return int(np.random.choice(ties))


class RoomEnv:
    """Grid room with one agent, one food, one poison."""

    def __init__(self):
        self.agent = None
        self.prev_agent = None
        self.food = None
        self.poison = None
        self.steps = 0
        self.reset()

    def reset(self):
        self.agent = (GRID // 2, GRID // 2)
        self.prev_agent = None
        self.food = random_empty({self.agent})
        self.poison = random_empty({self.agent, self.food})
        self.steps = 0
        return self.agent

    def step(self, action_idx: int):
        """One env step. Returns (next_state, reward, done)."""
        dx, dy = ACTIONS[action_idx]
        old_x, old_y = self.agent

        # Proposed move
        new_x = clamp(old_x + dx, 0, GRID - 1)
        new_y = clamp(old_y + dy, 0, GRID - 1)
        new_pos = (new_x, new_y)

        reward = REWARD_STEP
        done = False

        # Shaping: closer to food, farther from poison (Manhattan distance)
        old_food_dist = abs(old_x - self.food[0]) + abs(old_y - self.food[1])
        new_food_dist = abs(new_x - self.food[0]) + abs(new_y - self.food[1])

        old_pois_dist = abs(old_x - self.poison[0]) + abs(old_y - self.poison[1])
        new_pois_dist = abs(new_x - self.poison[0]) + abs(new_y - self.poison[1])

        reward += SHAPE_TO_FOOD * (old_food_dist - new_food_dist)
        reward += SHAPE_FROM_POISON * (new_pois_dist - old_pois_dist)

        # If hit wall (no movement), penalize
        if new_pos == (old_x, old_y):
            reward += REWARD_HIT_WALL

        # Update agent position
        self.agent = new_pos
        self.steps += 1

        # Backtrack penalty: if we undid the previous move exactly
        if self.prev_agent is not None and self.agent == self.prev_agent:
            reward += REWARD_BACKTRACK

        # Terminal rewards
        if self.agent == self.food:
            reward += REWARD_FOOD
            done = True
        elif self.agent == self.poison:
            reward += REWARD_POISON
            done = True

        if self.steps >= STEPS_PER_EP:
            done = True

        # Store previous position for next step
        self.prev_agent = (old_x, old_y)

        return self.agent, reward, done


class QAgent:
    """Tabular Q-learning agent on (x, y)."""

    def __init__(self):
        self.Q = np.zeros((GRID, GRID, N_ACTIONS), dtype=np.float32)

    def policy(self, state, eps: float) -> int:
        """Epsilon-greedy with random tie-breaking."""
        x, y = state
        if random.random() < eps:
            return random.randrange(N_ACTIONS)
        return greedy_action(self.Q[x, y])

    def update(self, s, a, r, s2):
        x, y = s
        nx, ny = s2
        best_next = np.max(self.Q[nx, ny])
        td_target = r + GAMMA * best_next
        td_error = td_target - self.Q[x, y, a]
        self.Q[x, y, a] += ALPHA * td_error


def train(agent: QAgent, env: RoomEnv):
    eps = EPS_START
    total_return = 0.0

    for ep in range(EPISODES_TRAIN):
        s = env.reset()
        G = 0.0
        for _ in range(STEPS_PER_EP):
            a = agent.policy(s, eps)
            s2, r, done = env.step(a)
            agent.update(s, a, r, s2)
            G += r
            s = s2
            if done:
                break

        total_return += G
        eps = max(EPS_END, eps * EPS_DECAY)

        if (ep + 1) % 100 == 0:
            avg = total_return / 100.0
            print(f"Episode {ep + 1:4d} | avg return (last 100): {avg:.3f} | eps {eps:.3f}")
            total_return = 0.0

    return agent


# ---------- Rendering ----------
def draw_circle(surface, grid_pos, color, radius=0.42):
    x, y = grid_pos
    cx = x * CELL + CELL // 2
    cy = y * CELL + CELL // 2
    rad = int(radius * CELL)
    pygame.draw.circle(surface, color, (cx, cy), rad)


def draw_square(surface, grid_pos, color):
    x, y = grid_pos
    rect = pygame.Rect(x * CELL, y * CELL, CELL, CELL)
    pygame.draw.rect(surface, color, rect)


def render_episode(agent: QAgent, env: RoomEnv, eps_play: float = 0.10, max_steps: int = STEPS_PER_EP):
    """Render one episode with slight exploration to avoid freezing/cycles."""
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    s = env.reset()
    total_r = 0.0

    for _ in range(max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        # Mild exploration during playback keeps it from 2-cycling
        a = agent.policy(s, eps=eps_play)
        s2, r, done = env.step(a)
        total_r += r

        # Draw
        screen.fill((22, 22, 28))
        for i in range(GRID + 1):
            pygame.draw.line(screen, (40, 40, 48), (i * CELL, 0), (i * CELL, H))
            pygame.draw.line(screen, (40, 40, 48), (0, i * CELL), (W, i * CELL))

        draw_circle(screen, env.poison, (200, 60, 60))
        draw_circle(screen, env.food, (60, 200, 100))
        draw_square(screen, env.agent, (240, 240, 240))

        pygame.display.flip()
        clock.tick(FPS_RENDER)

        s = s2
        if done:
            for _ in range(10):
                clock.tick(FPS_RENDER)
            break

    print(f"Rendered episode return: {total_r:.3f}")
    pygame.quit()


if __name__ == "__main__":
    env = RoomEnv()
    agent = QAgent()
    agent = train(agent, env)

    # Watch a few evaluation episodes
    for _ in range(3):
        render_episode(agent, env, eps_play=0.10)
