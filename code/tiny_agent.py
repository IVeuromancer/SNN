import pygame, sys, random, math, numpy as np

# ----------------------------
# Config
# ----------------------------
GRID = 20
CELL = 24
W = H = GRID * CELL
FPS_RENDER = 30

ALPHA = 0.2     # learning rate
GAMMA = 0.95    # discount
EPS_START = 1.0 # epsilon-greedy
EPS_END = 0.05
EPS_DECAY = 0.995

EPISODES_TRAIN = 800
STEPS_PER_EP = 200

REWARD_STEP = -0.01
REWARD_FOOD = +1.0
REWARD_POISON = -1.0

# ----------------------------
# Helpers
# ----------------------------
ACTIONS = [(0,1),(0,-1),(1,0),(-1,0)]  # down, up, right, left
N_ACTIONS = len(ACTIONS)

def clamp(v, lo, hi): return max(lo, min(hi, v))

def random_empty(exclude):
    """Return a random (x,y) not in exclude set."""
    while True:
        p = (random.randrange(GRID), random.randrange(GRID))
        if p not in exclude: return p

# ----------------------------
# Environment
# ----------------------------
class RoomEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.agent = (GRID//2, GRID//2)
        self.food = random_empty({self.agent})
        self.poison = random_empty({self.agent, self.food})
        self.steps = 0
        return self.agent

    def step(self, action_idx):
        dx, dy = ACTIONS[action_idx]
        x, y = self.agent
        x = clamp(x + dx, 0, GRID-1)
        y = clamp(y + dy, 0, GRID-1)
        self.agent = (x, y)
        self.steps += 1

        reward = REWARD_STEP
        done = False

        if self.agent == self.food:
            reward += REWARD_FOOD
            done = True
        elif self.agent == self.poison:
            reward += REWARD_POISON
            done = True

        if self.steps >= STEPS_PER_EP:
            done = True

        return self.agent, reward, done

# ----------------------------
# Q-Learning Agent (tabular)
# ----------------------------
class QAgent:
    def __init__(self):
        # Q-table shape: [GRID, GRID, N_ACTIONS]
        self.Q = np.zeros((GRID, GRID, N_ACTIONS), dtype=np.float32)

    def policy(self, state, eps):
        x, y = state
        if random.random() < eps:
            return random.randrange(N_ACTIONS)
        q = self.Q[x, y]
        return int(np.argmax(q))

    def update(self, s, a, r, s2):
        x, y = s
        nx, ny = s2
        best_next = np.max(self.Q[nx, ny])
        td_target = r + GAMMA * best_next
        td_error = td_target - self.Q[x, y, a]
        self.Q[x, y, a] += ALPHA * td_error

# ----------------------------
# Training (headless)
# ----------------------------
def train(agent, env):
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
            if done: break
        total_return += G
        eps = max(EPS_END, eps * EPS_DECAY)
        if (ep+1) % 100 == 0:
            avg = total_return / 100.0
            print(f"Episode {ep+1:4d} | avg return (last 100): {avg:.3f} | eps {eps:.3f}")
            total_return = 0.0
    return agent

# ----------------------------
# Rendering
# ----------------------------
def draw_circle(surface, grid_pos, color, radius=0.4):
    x, y = grid_pos
    cx = x * CELL + CELL // 2
    cy = y * CELL + CELL // 2
    rad = int(radius * CELL)
    pygame.draw.circle(surface, color, (cx, cy), rad)

def draw_square(surface, grid_pos, color):
    x, y = grid_pos
    rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
    pygame.draw.rect(surface, color, rect)

def render_episode(agent, env, greedy=True, max_steps=STEPS_PER_EP):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    s = env.reset()
    total_r = 0.0

    for step in range(max_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)

        # action
        if greedy:
            q = agent.Q[s[0], s[1]]
            a = int(np.argmax(q))
        else:
            a = agent.policy(s, eps=0.1)

        s2, r, done = env.step(a)
        total_r += r

        # draw
        screen.fill((22, 22, 28))
        # grid lines (purely cosmetic)
        for i in range(GRID+1):
            pygame.draw.line(screen, (40,40,48), (i*CELL,0), (i*CELL,H))
            pygame.draw.line(screen, (40,40,48), (0,i*CELL), (W,i*CELL))

        # poison, food, agent
        draw_circle(screen, env.poison, (200, 60, 60), radius=0.42)
        draw_circle(screen, env.food,   (60, 200, 100), radius=0.42)
        draw_square(screen, env.agent, (240,240,240))

        pygame.display.flip()
        clock.tick(FPS_RENDER)

        s = s2
        if done:
            # small pause to see terminal state
            for _ in range(10): 
                clock.tick(FPS_RENDER)
            break

    print(f"Rendered episode return: {total_r:.3f}")
    pygame.quit()

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    env = RoomEnv()
    agent = QAgent()
    agent = train(agent, env)

    # Watch a couple greedy rollouts
    for _ in range(3):
        render_episode(agent, env, greedy=True)
