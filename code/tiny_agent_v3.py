# tiny_agent_v_no_scent.py
# Gridworld Q-learning with stochastic policy and adjacent awareness (no scent fields).

import sys
import random
import numpy as np
import pygame
import os
import cv2

# ----------------------------
# Config
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GRID = 14
CELL = 28
W = H = GRID * CELL
FPS_RENDER = 30

# Q-learning
ALPHA = 0.25
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.12          # never fully zero to avoid freezing
EPS_DECAY = 0.995

# Stochastic policy knobs
TAU = 0.35              # softmax temperature (lower = greedier)
P_RANDOM_WALK = 0.08    # extra random step independent of epsilon

EPISODES_TRAIN = 3000
STEPS_PER_EP = 400

# Rewards
REWARD_STEP = -0.01
REWARD_FOOD = +1.0
REWARD_POISON = -1.0
REWARD_BACKTRACK = -0.02
REWARD_HIT_WALL = -0.02

# Actions: down, up, right, left
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
N_ACTIONS = len(ACTIONS)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def random_empty(exclude):
    while True:
        p = (random.randrange(GRID), random.randrange(GRID))
        if p not in exclude:
            return p


def softmax_action(qrow: np.ndarray, tau: float) -> int:
    z = qrow - np.max(qrow)
    p = np.exp(z / max(tau, 1e-6))
    p /= p.sum()
    return int(np.random.choice(len(qrow), p=p))


def greedy_action_tiebreak(qrow: np.ndarray) -> int:
    mx = np.max(qrow)
    ties = np.flatnonzero(qrow == mx)
    return int(np.random.choice(ties))


class RoomEnv:
    """Grid room with one agent, one food, one poison. Agent gets no global scent, but can detect adjacent targets."""
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

    def neighbors(self, pos):
        """Return dict mapping action_idx -> neighbor_pos for valid in-bounds neighbors."""
        x, y = pos
        nbs = {}
        for a_idx, (dx, dy) in enumerate(ACTIONS):
            nx, ny = clamp(x + dx, 0, GRID - 1), clamp(y + dy, 0, GRID - 1)
            nbs[a_idx] = (nx, ny)
        return nbs

    def step(self, action_idx: int):
        dx, dy = ACTIONS[action_idx]
        old_x, old_y = self.agent
        new_x = clamp(old_x + dx, 0, GRID - 1)
        new_y = clamp(old_y + dy, 0, GRID - 1)
        new_pos = (new_x, new_y)

        reward = REWARD_STEP
        done = False

        # Hitting wall (no movement) penalty
        if new_pos == (old_x, old_y):
            reward += REWARD_HIT_WALL

        # Commit move
        self.agent = new_pos
        self.steps += 1

        # Backtrack penalty: undo last move
        if self.prev_agent is not None and self.agent == self.prev_agent:
            reward += REWARD_BACKTRACK

        # Terminals
        if self.agent == self.food:
            reward += REWARD_FOOD
            done = True
        elif self.agent == self.poison:
            reward += REWARD_POISON
            done = True

        if self.steps >= STEPS_PER_EP:
            done = True

        self.prev_agent = (old_x, old_y)
        return self.agent, reward, done


class QAgent:
    """Tabular Q-learning on (x, y) with adjacent awareness.
       Reflex layer: if food is adjacent, go for it; mask out actions into adjacent poison when sampling."""
    def __init__(self, env: RoomEnv):
        self.Q = np.zeros((GRID, GRID, N_ACTIONS), dtype=np.float32)
        self.env = env

    def policy(self, state, eps: float) -> int:
        x, y = state

        # 1) Hard-coded reflex: if food is adjacent, take it immediately.
        nbs = self.env.neighbors((x, y))
        for a_idx, pos in nbs.items():
            if pos == self.env.food:
                return a_idx  # go eat

        # 2) Optional pure random-walk to keep things lively
        if random.random() < P_RANDOM_WALK:
            # avoid stepping into adjacent poison if possible
            safe_actions = [a for a, p in nbs.items() if p != self.env.poison]
            if safe_actions:
                return random.choice(safe_actions)
            return random.randrange(N_ACTIONS)

        # 3) Epsilon-greedy exploration
        if random.random() < eps:
            safe_actions = [a for a, p in nbs.items() if p != self.env.poison]
            if safe_actions:
                return random.choice(safe_actions)
            return random.randrange(N_ACTIONS)

        # 4) Softmax over Q, but mask actions that step into adjacent poison
        qrow = self.Q[x, y].copy()
        for a_idx, pos in nbs.items():
            if pos == self.env.poison:
                qrow[a_idx] = -1e9  # effectively remove from softmax
        # If all got masked (rare), fall back to greedy tie-break
        if np.all(qrow == -1e9):
            return greedy_action_tiebreak(self.Q[x, y])
        return softmax_action(qrow, TAU)

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
            print(f"Episode {ep + 1:4d} | avg return (last 100): {total_return / 100:.3f} | eps {eps:.3f}")
            total_return = 0.0
    return agent


# ---------- Rendering ----------
def draw_circle(surface, grid_pos, color, radius=0.42):
    x, y = grid_pos
    cx = x * CELL + CELL // 2
    cy = y * CELL + CELL // 2
    pygame.draw.circle(surface, color, (cx, cy), int(radius * CELL))


def draw_square(surface, grid_pos, color):
    x, y = grid_pos
    pygame.draw.rect(surface, color, pygame.Rect(x * CELL, y * CELL, CELL, CELL))


def render_multiple_episodes(agent: QAgent, env: RoomEnv, n_episodes=5,
                             eps_play=0.15, max_steps=10000, video_name="agent_run.mp4"):
    """Render several episodes in one continuous video."""
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_name, fourcc, FPS_RENDER, (W, H))

    for ep in range(n_episodes):
        s = env.reset()
        total_r = 0.0
        print(f"Starting episode {ep+1}/{n_episodes}")

        for _ in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    out.release()
                    pygame.quit()
                    sys.exit(0)

            a = agent.policy(s, eps_play)
            s2, r, done = env.step(a)
            total_r += r

            # draw
            screen.fill((22, 22, 28))
            for i in range(GRID + 1):
                pygame.draw.line(screen, (40, 40, 48), (i * CELL, 0), (i * CELL, H))
                pygame.draw.line(screen, (40, 40, 48), (0, i * CELL), (W, i * CELL))

            draw_circle(screen, env.poison, (200, 60, 60))
            draw_circle(screen, env.food,   (60, 200, 100))
            draw_square(screen, env.agent,  (240, 240, 240))

            # write frame to video
            frame = pygame.surfarray.array3d(screen)
            frame = np.rot90(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

            pygame.display.flip()
            clock.tick(FPS_RENDER)

            s = s2
            if done:
                break

        print(f"Episode {ep+1} return: {total_r:.3f}")

    out.release()
    pygame.quit()
    print(f"Saved combined video to {video_name}")


if __name__ == "__main__":
    env = RoomEnv()
    agent = QAgent(env)

    # --- Training only if no saved model ---
    if os.path.exists("q_table.npy"):
        print("Loading existing Q-table...")
        agent.Q = np.load("q_table.npy")
    else:
        print("Training new agent...")
        agent = train(agent, env)
        np.save("q_table.npy", agent.Q)
        print("Training complete and saved to q_table.npy")

    # --- Evaluation ---
    render_multiple_episodes(agent, env,
                            n_episodes=5,
                            eps_play=0.15,
                            max_steps=10000,
                            video_name="agent_run.mp4")
