"""
Pong Reinforcement Learning Laboratory (Unified Edition)
-------------------------------------------------------
This is a single-file interactive educational laboratory designed to teach
Reinforcement Learning concepts from scratch.

Modes:
1. Train Mode:
   - Headless (default): python pong_rl_lab.py --train
   - Visual: python pong_rl_lab.py --train --render
2. Replay Mode:
   - python pong_rl_lab.py --replay checkpoints/qtable_500.pkl
3. Analyze Mode:
   - python pong_rl_lab.py --analyze checkpoints/qtable_1000.pkl
4. Resume Mode:
   - python pong_rl_lab.py --resume checkpoints/qtable_500.pkl [--render]
"""

import sys
import os
import pickle
import random
import time
import argparse
from datetime import datetime
import numpy as np
import pygame

# Ensure checkpoints folder exists
os.makedirs("checkpoints", exist_ok=True)

# ── Screen Configuration ──────────────────────────────────────────
WIDTH, HEIGHT = 1200, 700
SIDEBAR_WIDTH = 320
FPS = 60

# ── Color Palette (Catppuccin Mocha Inspired) ─────────────────────
COLOR_BG = (24, 24, 37)          # Dark Slate Blue
COLOR_SIDEBAR_BG = (17, 17, 27)  # Deep Charcoal
COLOR_PADDLE = (137, 220, 235)    # Neon Cyan
COLOR_BALL = (245, 194, 231)      # Soft Pink
COLOR_WALLS = (88, 91, 112)       # Slate Gray
COLOR_TEXT_MAIN = (205, 214, 244) # Off-White
COLOR_TEXT_STATE = (166, 227, 161)# Pastel Green
COLOR_CARD = (30, 30, 46)         # Deep Card Navy
COLOR_RED = (243, 139, 168)       # Neon Red (worst action / miss)
COLOR_ORANGE = (250, 179, 135)    # Warm Orange
COLOR_YELLOW = (249, 226, 175)    # Pastel Yellow
COLOR_CYAN = (137, 180, 250)      # Sky Blue

# Flash Colors
COLOR_FLASH_GREEN = (166, 227, 161)
COLOR_FLASH_RED = (243, 139, 168)


# ── Action Definition ─────────────────────────────────────────────
class Action:
    STAY = 0
    UP = 1
    DOWN = 2
    NAMES = {STAY: "STAY", UP: "UP", DOWN: "DOWN"}


# ── Q-Learning Agent ──────────────────────────────────────────────
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, min_epsilon=0.05, decay_rate=0.995):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(3)  # [STAY, UP, DOWN]
        return self.q_table[state]

    def choose_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.choice([Action.STAY, Action.UP, Action.DOWN])
        
        q_vals = self.get_q_values(state)
        max_val = np.max(q_vals)
        ties = np.where(q_vals == max_val)[0]
        return int(random.choice(ties))

    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q_values(state)[action]
        next_max = 0.0 if done else np.max(self.get_q_values(next_state))
        new_q = current_q + self.alpha * (reward + self.gamma * next_max - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)


# ── Environment Entities ──────────────────────────────────────────
class Ball:
    def __init__(self):
        self.radius = 12
        self.reset()

    def reset(self):
        self.x = SIDEBAR_WIDTH + (WIDTH - SIDEBAR_WIDTH) // 2
        self.y = HEIGHT // 2
        self.vx = random.choice([-6, -5, 5, 6])
        self.vy = random.choice([-4, -3, 3, 4])

    def update(self):
        self.x += self.vx
        self.y += self.vy

        # Wall bounces (top/bottom)
        if self.y - self.radius <= 0:
            self.y = self.radius
            self.vy = -self.vy
        elif self.y + self.radius >= HEIGHT:
            self.y = HEIGHT - self.radius
            self.vy = -self.vy

        # Arena boundary bounce (left edge of playable arena)
        if self.x - self.radius <= SIDEBAR_WIDTH:
            self.x = SIDEBAR_WIDTH + self.radius
            self.vx = -self.vx

    def draw(self, screen):
        pygame.draw.circle(screen, COLOR_BALL, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius - 3, 2)


class Paddle:
    def __init__(self):
        self.width = 16
        self.height = 100
        self.reset()
        self.speed = 18

    def reset(self):
        self.x = WIDTH - 40
        self.y = HEIGHT // 2 - self.height // 2

    def move_up(self):
        self.y = max(0, self.y - self.speed)

    def move_down(self):
        self.y = min(HEIGHT - self.height, self.y + self.speed)

    def draw(self, screen):
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, COLOR_PADDLE, rect, border_radius=8)


class FloatingText:
    def __init__(self, x, y, text, color, duration=45):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.timer = duration
        self.max_timer = duration

    def update(self):
        self.y -= 1.2
        self.timer -= 1

    def draw(self, screen, font):
        alpha = int((self.timer / self.max_timer) * 255)
        text_surf = font.render(self.text, True, self.color)
        temp_surf = pygame.Surface(text_surf.get_size(), pygame.SRCALPHA)
        temp_surf.blit(text_surf, (0, 0))
        temp_surf.set_alpha(alpha)
        rect = temp_surf.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(temp_surf, rect)


# ── Environment ───────────────────────────────────────────────────
class Environment:
    def __init__(self, state_mode="advanced"):
        self.state_mode = state_mode
        self.ball = Ball()
        self.paddle = Paddle()
        
        # Statistics & Reward Tracking
        self.hits = 0
        self.misses = 0
        self.episode = 1
        
        self.current_reward = 0
        self.episode_reward = 0
        self.total_reward = 0
        
        self.episode_history = []
        self.best_reward = -9999.0
        self.best_hit_rate = 0.0

        # Visual FX Hooks
        self.flash_timer = 0
        self.flash_max = 12
        self.flash_color = COLOR_FLASH_GREEN
        self.floating_texts = []

    def get_state(self):
        # Discretize y positions into 10 vertical zones
        ball_y_zone = max(0, min(9, int(self.ball.y / (HEIGHT / 10))))
        paddle_center = self.paddle.y + self.paddle.height // 2
        paddle_y_zone = max(0, min(9, int(paddle_center / (HEIGHT / 10))))
        
        # Directions (UP = 0, DOWN = 1, FLAT = 2)
        if self.ball.vy < 0:
            ball_vy_direction = 0
        elif self.ball.vy > 0:
            ball_vy_direction = 1
        else:
            ball_vy_direction = 2
            
        ball_vx_direction = 0 if self.ball.vx < 0 else 1

        if self.state_mode == "basic":
            return (ball_y_zone, paddle_y_zone, ball_vy_direction, ball_vx_direction)
        else:
            # Advanced mode: adds ball_x_zone discretized into 10 bins
            arena_width = WIDTH - SIDEBAR_WIDTH
            ball_x_zone = max(0, min(9, int((self.ball.x - SIDEBAR_WIDTH) / (arena_width / 10))))
            return (ball_x_zone, ball_y_zone, paddle_y_zone, ball_vy_direction, ball_vx_direction)

    def step(self, action):
        reward = 0
        done = False

        if action == Action.UP:
            self.paddle.move_up()
        elif action == Action.DOWN:
            self.paddle.move_down()

        self.ball.update()

        # Paddle collision check (+1 Reward)
        if (self.ball.vx > 0 and 
            self.paddle.x <= self.ball.x + self.ball.radius <= self.paddle.x + self.paddle.width):
            if (self.paddle.y <= self.ball.y <= self.paddle.y + self.paddle.height):
                self.ball.vx = -abs(self.ball.vx)
                paddle_center = self.paddle.y + self.paddle.height / 2
                hit_diff = self.ball.y - paddle_center
                self.ball.vy += int(hit_diff * 0.1)
                self.ball.vx = max(-10, min(10, self.ball.vx))
                self.ball.vy = max(-10, min(10, self.ball.vy))
                
                reward = 1
                self.hits += 1
                
                self.flash_color = COLOR_FLASH_GREEN
                self.flash_timer = self.flash_max
                self.floating_texts.append(
                    FloatingText(self.ball.x - 20, self.ball.y, "+1", COLOR_FLASH_GREEN)
                )

        # Miss check (-10 Reward, Episode Terminal)
        if self.ball.x - self.ball.radius > WIDTH:
            reward = -10
            self.misses += 1
            done = True
            
            self.flash_color = COLOR_FLASH_RED
            self.flash_timer = self.flash_max
            self.floating_texts.append(
                FloatingText(WIDTH - 50, self.ball.y, "-10", COLOR_FLASH_RED, duration=60)
            )

        self.current_reward = reward
        self.episode_reward += reward
        self.total_reward += reward

        if done:
            self.episode_history.append(self.episode_reward)
            if self.episode_reward > self.best_reward:
                self.best_reward = self.episode_reward
                
            total_attempts = self.hits + self.misses
            current_hr = (self.hits / total_attempts * 100) if total_attempts > 0 else 0.0
            if current_hr > self.best_hit_rate:
                self.best_hit_rate = current_hr
                
            self.episode += 1
            self.episode_reward = 0
            self.ball.reset()
            self.paddle.reset()

        return self.get_state(), reward, done


# ── UI Elements & Helper Panels ───────────────────────────────────
def draw_card(screen, rect, title, font):
    pygame.draw.rect(screen, COLOR_CARD, rect, border_radius=8)
    pygame.draw.rect(screen, COLOR_WALLS, rect, 2, border_radius=8)
    if title:
        screen.blit(font.render(title, True, COLOR_ORANGE), (rect.x + 15, rect.y + 10))


def draw_mini_graph(screen, data, rect, color, label, font):
    pygame.draw.rect(screen, (20, 20, 30), rect, border_radius=4)
    pygame.draw.rect(screen, COLOR_WALLS, rect, 1, border_radius=4)
    
    if len(data) < 2:
        screen.blit(font.render(f"{label} (needs data)", True, COLOR_WALLS), (rect.x + 10, rect.y + 10))
        return

    pts = min(len(data), rect.width - 10)
    step = max(1, len(data) // pts)
    sampled = data[::step][-pts:]
    mn, mx = min(sampled), max(sampled)
    rng = (mx - mn) or 1.0

    points = []
    for i, v in enumerate(sampled):
        px = rect.x + 5 + int(i * (rect.width - 10) / len(sampled))
        py = rect.y + rect.height - 5 - int((v - mn) / rng * (rect.height - 10))
        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(screen, color, False, points, 2)

    screen.blit(font.render(label, True, COLOR_TEXT_MAIN), (rect.x + 5, rect.y + 5))


def draw_inspector(screen, state, agent, font, large_font):
    """Draws visual Q-table details overlaying the game area."""
    overlay = pygame.Surface((400, 250), pygame.SRCALPHA)
    overlay.fill((30, 30, 46, 230))  # Glassmorphism dark background
    pygame.draw.rect(overlay, COLOR_ORANGE, (0, 0, 400, 250), 3, border_radius=10)

    title = large_font.render("Q-TABLE INSPECTOR", True, COLOR_ORANGE)
    state_str = font.render(f"State: {state}", True, COLOR_TEXT_STATE)
    overlay.blit(title, (20, 20))
    overlay.blit(state_str, (20, 55))

    q_vals = agent.get_q_values(state)
    max_idx = np.argmax(q_vals)
    all_zero = np.all(q_vals == 0.0)

    for idx, act in enumerate(["STAY", "UP", "DOWN"]):
        val = q_vals[idx]
        is_best = (idx == max_idx and not all_zero)
        color = COLOR_TEXT_STATE if is_best else (COLOR_RED if val < 0 else COLOR_TEXT_MAIN)
        txt = font.render(f"Q({act}): {val:+.4f}", True, color)
        overlay.blit(txt, (40, 100 + idx * 35))
        if is_best:
            pygame.draw.circle(overlay, COLOR_TEXT_STATE, (25, 110 + idx * 35), 6)

    screen.blit(overlay, (SIDEBAR_WIDTH + (WIDTH - SIDEBAR_WIDTH)//2 - 200, HEIGHT//2 - 125))


def draw_heatmap(screen, agent, state_mode, font, large_font):
    """Draws the 10x10 state exploration grid overlay."""
    overlay = pygame.Surface((500, 450), pygame.SRCALPHA)
    overlay.fill((17, 17, 27, 240))
    pygame.draw.rect(overlay, COLOR_CYAN, (0, 0, 500, 450), 3, border_radius=10)

    title = large_font.render("EXPLORATION HEATMAP (Ball Y vs Paddle Y)", True, COLOR_CYAN)
    overlay.blit(title, (20, 20))

    cell_size = 28
    start_x = 110
    start_y = 70

    # Draw Y axis label
    y_lbl = font.render("Ball Y Zone (0-9)", True, COLOR_ORANGE)
    y_lbl_rot = pygame.transform.rotate(y_lbl, 90)
    overlay.blit(y_lbl_rot, (25, start_y + 80))

    # Draw X axis label
    x_lbl = font.render("Paddle Y Zone (0-9)", True, COLOR_ORANGE)
    overlay.blit(x_lbl, (start_x + 60, start_y + cell_size * 10 + 35))

    # Axis Numbers
    for i in range(10):
        overlay.blit(font.render(str(i), True, COLOR_TEXT_MAIN), (85, start_y + i * cell_size + 5))
        overlay.blit(font.render(str(i), True, COLOR_TEXT_MAIN), (start_x + i * cell_size + 8, start_y - 20))

    # Check state visits
    visited_cells = 0
    for byz in range(10):
        for pyz in range(10):
            match_found = False
            for k in agent.q_table.keys():
                if len(k) == 5:
                    if k[1] == byz and k[2] == pyz:
                        match_found = True
                        break
                elif len(k) == 4:
                    if k[0] == byz and k[1] == pyz:
                        match_found = True
                        break

            rect = pygame.Rect(start_x + pyz * cell_size, start_y + byz * cell_size, cell_size - 2, cell_size - 2)
            if match_found:
                pygame.draw.rect(overlay, COLOR_TEXT_STATE, rect, border_radius=3)
                visited_cells += 1
            else:
                pygame.draw.rect(overlay, (40, 40, 55), rect, border_radius=3)

    summary_str = font.render(f"Visited Coordinates: {visited_cells}/100 pairs", True, COLOR_TEXT_MAIN)
    overlay.blit(summary_str, (110, start_y + cell_size * 10 + 10))

    screen.blit(overlay, (SIDEBAR_WIDTH + (WIDTH - SIDEBAR_WIDTH)//2 - 250, HEIGHT//2 - 225))


def draw_educational(screen, font, large_font):
    """Draws educational explanation guide panel."""
    overlay = pygame.Surface((650, 520), pygame.SRCALPHA)
    overlay.fill((24, 24, 37, 245))
    pygame.draw.rect(overlay, COLOR_YELLOW, (0, 0, 650, 520), 3, border_radius=12)

    title = large_font.render("REINFORCEMENT LEARNING LAB GUIDE", True, COLOR_YELLOW)
    overlay.blit(title, (30, 25))

    concepts = [
        ("State (S)", "The information describing the env. Basic: 600 states (10 Y * 10 P * 3 vy * 2 vx). Advanced: 6000 states (adds 10 X)."),
        ("Action (A)", "The decisions available to the agent. At every time step, it chooses to STAY, move UP, or move DOWN."),
        ("Reward (R)", "Feedback signals sent from the environment. Successfully blocking the ball yields +1. Letting the ball pass yields -10."),
        ("Q-Value Q(s,a)", "Represents the Quality (expected cumulative future reward) of taking a specific action 'a' in state 's'."),
        ("Epsilon (E)", "Exploration coefficient. Probability of selecting a random action. Epsilon decays as the agent learns, transitioning to pure exploitation."),
        ("Bellman Update", "Formula used to adjust values based on Temporal Difference errors: Q(s,a) = Q(s,a) + alpha * [Reward + gamma * max Q(s',a') - Q(s,a)]")
    ]

    for idx, (name, desc) in enumerate(concepts):
        header = font.render(f"■ {name}", True, COLOR_ORANGE)
        overlay.blit(header, (30, 75 + idx * 72))

        # Wrap text manually
        words = desc.split(' ')
        lines = []
        curr_line = ""
        for w in words:
            if font.size(curr_line + w)[0] < 580:
                curr_line += w + " "
            else:
                lines.append(curr_line)
                curr_line = w + " "
        lines.append(curr_line)

        for l_idx, line in enumerate(lines):
            t_surf = font.render(line, True, COLOR_TEXT_MAIN)
            overlay.blit(t_surf, (50, 95 + idx * 72 + l_idx * 18))

    screen.blit(overlay, (SIDEBAR_WIDTH + (WIDTH - SIDEBAR_WIDTH)//2 - 325, HEIGHT//2 - 260))


# ── Mode Executions ───────────────────────────────────────────────

def analyze_checkpoint(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "q_table" in data:
        q_table = data["q_table"]
    else:
        q_table = data

    num_states = len(q_table)
    if num_states == 0:
        print("Empty Q-Table.")
        return

    # Calculate Q-value stats
    all_qs = np.array(list(q_table.values()))
    min_q = np.min(all_qs)
    max_q = np.max(all_qs)
    mean_q = np.mean(all_qs)

    # Find best state and worst state
    best_state = None
    worst_state = None
    best_val = -float('inf')
    worst_val = float('inf')

    for state, qs in q_table.items():
        state_max = np.max(qs)
        state_min = np.min(qs)
        if state_max > best_val:
            best_val = state_max
            best_state = state
        if state_min < worst_val:
            worst_val = state_min
            worst_state = state

    print("=" * 60)
    print(f"             Q-TABLE ANALYSIS REPORT: {os.path.basename(ckpt_path)}")
    print("=" * 60)
    print(f"Q-table size:             {num_states}")
    print(f"Number of visited states: {num_states}")
    print(f"Best state:               {best_state} (Max Q: {best_val:+.4f})")
    print(f"Worst state:              {worst_state} (Min Q: {worst_val:+.4f})")
    print(f"Average Q-value:          {mean_q:+.4f}")
    print(f"Max Q-value:              {max_q:+.4f}")
    print(f"Min Q-value:              {min_q:+.4f}")
    print("=" * 60)


def replay_mode(ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)

    is_compatible = False
    if isinstance(data, dict):
        q_table = data.get("q_table", {})
        if data.get("version") == "ternary_vy":
            is_compatible = True
    else:
        q_table = data

    # Infer state dimensions
    if len(q_table) > 0:
        first_key = list(q_table.keys())[0]
        state_mode = "basic" if len(first_key) == 4 else "advanced"
    else:
        state_mode = "advanced"

    if not is_compatible:
        print("=" * 80)
        print(" [WARNING] COMPATIBILITY ALERT")
        print(" The loaded checkpoint does not match the new state representation (ternary vy).")
        print(" Loading old checkpoints (UP/DOWN only) will lead to state aliasing and bugs.")
        print("=" * 80)

    print(f"Replaying checkpoint '{ckpt_path}' in '{state_mode}' mode.")

    # Initialize Pygame explicitly before font operations
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Pong RL Lab: Replay Mode — {os.path.basename(ckpt_path)}")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("Outfit", 18, bold=True)
        small_font = pygame.font.SysFont("Outfit", 14, bold=True)
        large_font = pygame.font.SysFont("Outfit", 22, bold=True)
    except Exception:
        font = pygame.font.Font(None, 22)
        small_font = pygame.font.Font(None, 16)
        large_font = pygame.font.Font(None, 26)

    agent = QLearningAgent(epsilon=0.0)
    agent.q_table = q_table

    env = Environment(state_mode=state_mode)
    running = True

    show_inspector = False
    show_heatmap = False
    show_educational = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_i:
                    show_inspector = not show_inspector
                    show_heatmap = False; show_educational = False
                elif event.key == pygame.K_h:
                    show_heatmap = not show_heatmap
                    show_inspector = False; show_educational = False
                elif event.key == pygame.K_e:
                    show_educational = not show_educational
                    show_inspector = False; show_heatmap = False

        state = env.get_state()
        action = agent.choose_action(state, explore=False)
        next_state, reward, done = env.step(action)

        if done:
            env.ball.reset()
            env.paddle.reset()

        # Render
        screen.fill(COLOR_BG)
        pygame.draw.rect(screen, COLOR_SIDEBAR_BG, (0, 0, SIDEBAR_WIDTH, HEIGHT))
        pygame.draw.line(screen, COLOR_WALLS, (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, HEIGHT), 3)

        arena_center_x = SIDEBAR_WIDTH + (WIDTH - SIDEBAR_WIDTH) // 2
        pygame.draw.line(screen, COLOR_WALLS, (arena_center_x, 0), (arena_center_x, HEIGHT), 2)

        env.ball.draw(screen)
        env.paddle.draw(screen)

        # Screen Flash FX
        if env.flash_timer > 0:
            flash_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            alpha = int((env.flash_timer / env.flash_max) * 80)
            flash_surface.fill((*env.flash_color, alpha))
            screen.blit(flash_surface, (0, 0))
            env.flash_timer -= 1

        # Sidebar HUD
        screen.blit(large_font.render("REPLAY MODE", True, COLOR_CYAN), (20, 20))
        screen.blit(small_font.render(f"File: {os.path.basename(ckpt_path)}", True, COLOR_ORANGE), (20, 48))

        # Dashboard Card
        c1 = pygame.Rect(20, 80, 280, 160)
        draw_card(screen, c1, "REPLAY HUD STATS", font)
        total_a = env.hits + env.misses
        hit_rate = (env.hits / total_a * 100) if total_a > 0 else 0.0
        screen.blit(font.render(f"States Loaded: {len(q_table)}", True, COLOR_TEXT_STATE), (35, 115))
        screen.blit(font.render(f"Epsilon: 0.0000 (exploit)", True, COLOR_TEXT_MAIN), (35, 138))
        screen.blit(font.render(f"Hit Rate: {hit_rate:.1f}%", True, COLOR_YELLOW), (35, 161))
        screen.blit(font.render(f"Hits: {env.hits} | Misses: {env.misses}", True, COLOR_TEXT_MAIN), (35, 184))
        screen.blit(font.render(f"State Mode: {state_mode.upper()}", True, COLOR_CYAN), (35, 207))

        # Telemetry Card
        c2 = pygame.Rect(20, 255, 280, 140)
        draw_card(screen, c2, "TELEMETRY", font)
        screen.blit(small_font.render(f"STATE: {state}", True, COLOR_TEXT_STATE), (35, 280))
        
        # Unpack state coordinates for human-readable direction labels
        vy_val = state[3] if len(state) == 5 else state[2]
        vx_val = state[4] if len(state) == 5 else state[3]
        vy_lbl = "UP" if vy_val == 0 else ("DOWN" if vy_val == 1 else "FLAT")
        vx_lbl = "LEFT" if vx_val == 0 else "RIGHT"
        
        screen.blit(font.render(f"Ball dY: {vy_lbl} | dX: {vx_lbl}", True, COLOR_TEXT_MAIN), (35, 305))
        screen.blit(font.render(f"ACTION: {Action.NAMES[action]}", True, COLOR_TEXT_MAIN), (35, 330))

        # Render action arrow indicators
        indicator_x = 260
        indicator_y = 330
        if action == Action.UP:
            pygame.draw.polygon(screen, COLOR_TEXT_STATE, [(indicator_x, indicator_y - 12), (indicator_x - 12, indicator_y + 6), (indicator_x + 12, indicator_y + 6)])
        elif action == Action.DOWN:
            pygame.draw.polygon(screen, COLOR_TEXT_STATE, [(indicator_x, indicator_y + 12), (indicator_x - 12, indicator_y - 6), (indicator_x + 12, indicator_y - 6)])
        else:
            pygame.draw.rect(screen, COLOR_ORANGE, (indicator_x - 10, indicator_y - 10, 20, 20), border_radius=3)

        # Values Card
        c3 = pygame.Rect(20, 410, 280, 140)
        draw_card(screen, c3, "Q-VALUES", font)
        q_vals = agent.get_q_values(state)
        for idx, act in enumerate(["STAY", "UP", "DOWN"]):
            marker = " ◄" if idx == action else ""
            screen.blit(font.render(f"Q({act}): {q_vals[idx]:+.4f}{marker}", True, COLOR_TEXT_MAIN), (35, 440 + idx * 26))

        # Instructions Footer
        screen.blit(font.render("[I] Inspect Q-values", True, COLOR_CYAN), (35, 570))
        screen.blit(font.render("[H] Visited Heatmap", True, COLOR_CYAN), (35, 595))
        screen.blit(font.render("[E] Educational Guide", True, COLOR_CYAN), (35, 620))
        screen.blit(font.render("[ESC] Exit replay", True, COLOR_WALLS), (35, 655))

        # Draw overlays
        if show_inspector:
            draw_inspector(screen, state, agent, font, large_font)
        elif show_heatmap:
            draw_heatmap(screen, agent, state_mode, font, large_font)
        elif show_educational:
            draw_educational(screen, font, large_font)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def generate_replay_gif(agent, state_mode, filepath):
    import os
    import numpy as np
    from PIL import Image
    
    old_driver = os.environ.get("SDL_VIDEODRIVER")
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    pygame.init()
    surf = pygame.Surface((WIDTH, HEIGHT))
    
    # Initialize a temporary environment
    temp_env = Environment(state_mode=state_mode)
    
    frames = []
    
    try:
        gif_font = pygame.font.SysFont("Outfit", 18, bold=True)
        gif_small = pygame.font.SysFont("Outfit", 14, bold=True)
    except Exception:
        gif_font = pygame.font.Font(None, 22)
        gif_small = pygame.font.Font(None, 16)
        
    state = temp_env.get_state()
    for _ in range(150):
        # Epsilon = 0 (exploitation only to show what was learned)
        q_vals = agent.get_q_values(state)
        action = int(np.argmax(q_vals))
        
        next_state, reward, done = temp_env.step(action)
        state = next_state
        
        # Render the state on surf
        surf.fill(COLOR_BG)
        pygame.draw.rect(surf, COLOR_SIDEBAR_BG, (0, 0, SIDEBAR_WIDTH, HEIGHT))
        pygame.draw.line(surf, COLOR_WALLS, (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, HEIGHT), 3)
        arena_center_x = SIDEBAR_WIDTH + (WIDTH - SIDEBAR_WIDTH) // 2
        pygame.draw.line(surf, COLOR_WALLS, (arena_center_x, 0), (arena_center_x, HEIGHT), 2)
        
        temp_env.ball.draw(surf)
        temp_env.paddle.draw(surf)
        
        # Cards
        c1 = pygame.Rect(20, 80, 280, 160)
        draw_card(surf, c1, "REPLAY HUD STATS", gif_font)
        total_a = temp_env.hits + temp_env.misses
        hit_rate = (temp_env.hits / total_a * 100) if total_a > 0 else 0.0
        surf.blit(gif_font.render(f"States Loaded: {len(agent.q_table)}", True, COLOR_TEXT_STATE), (35, 115))
        surf.blit(gif_font.render(f"Epsilon: 0.0000 (exploit)", True, COLOR_TEXT_MAIN), (35, 138))
        surf.blit(gif_font.render(f"Hit Rate: {hit_rate:.1f}%", True, COLOR_YELLOW), (35, 161))
        surf.blit(gif_font.render(f"Hits: {temp_env.hits} | Misses: {temp_env.misses}", True, COLOR_TEXT_MAIN), (35, 184))
        surf.blit(gif_font.render(f"State Mode: {state_mode.upper()}", True, COLOR_CYAN), (35, 207))
        
        c2 = pygame.Rect(20, 255, 280, 140)
        draw_card(surf, c2, "TELEMETRY", gif_font)
        surf.blit(gif_small.render(f"STATE: {state}", True, COLOR_TEXT_STATE), (35, 280))
        vy_val = state[3] if len(state) == 5 else state[2]
        vx_val = state[4] if len(state) == 5 else state[3]
        vy_lbl = "UP" if vy_val == 0 else ("DOWN" if vy_val == 1 else "FLAT")
        vx_lbl = "LEFT" if vx_val == 0 else "RIGHT"
        surf.blit(gif_font.render(f"Ball dY: {vy_lbl} | dX: {vx_lbl}", True, COLOR_TEXT_MAIN), (35, 305))
        surf.blit(gif_font.render(f"ACTION: {Action.NAMES[action]}", True, COLOR_TEXT_MAIN), (35, 330))
        
        c3 = pygame.Rect(20, 410, 280, 140)
        draw_card(surf, c3, "Q-VALUES", gif_font)
        for idx, act in enumerate(["STAY", "UP", "DOWN"]):
            marker = " ◄" if idx == action else ""
            surf.blit(gif_font.render(f"Q({act}): {q_vals[idx]:+.4f}{marker}", True, COLOR_TEXT_MAIN), (35, 440 + idx * 26))
            
        surf.blit(gif_font.render("RECORDING REPLAY...", True, COLOR_RED), (20, 20))
        
        # Convert Pygame Surface to PIL Image and scale down
        scaled_surf = pygame.transform.smoothscale(surf, (480, 280))
        img_str = pygame.image.tostring(scaled_surf, "RGB")
        img = Image.frombytes("RGB", (480, 280), img_str)
        frames.append(img)
        
        if done:
            temp_env.ball.reset()
            temp_env.paddle.reset()
            state = temp_env.get_state()
            
    # Save the frames as an animated GIF
    if frames:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        frames[0].save(
            filepath,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=50,  # 50ms per frame = 20fps
            loop=0
        )
        
    if old_driver is not None:
        os.environ["SDL_VIDEODRIVER"] = old_driver
    else:
        del os.environ["SDL_VIDEODRIVER"]


def save_replays_and_stats(agent, env, completed_episode, avg_reward, hit_rate):
    import json
    import os
    
    # 1. Update stats.json
    stats_dir = "website/data"
    os.makedirs(stats_dir, exist_ok=True)
    
    checkpoints = []
    if os.path.exists("checkpoints"):
        checkpoints = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pkl")])
        
    stats = {
        "episode": int(completed_episode),
        "hit_rate": float(hit_rate),
        "epsilon": float(agent.epsilon),
        "qtable_size": int(len(agent.q_table)),
        "average_reward": float(avg_reward),
        "checkpoints": checkpoints
    }
    
    with open(os.path.join(stats_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
        
    # 2. Update history.json
    history_file = os.path.join(stats_dir, "history.json")
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
                if not isinstance(history, list):
                    history = []
        except Exception:
            history = []
            
    history = [item for item in history if item.get("episode") != completed_episode]
    history.append({
        "episode": int(completed_episode),
        "hit_rate": float(hit_rate),
        "epsilon": float(agent.epsilon),
        "qtable_size": int(len(agent.q_table)),
        "average_reward": float(avg_reward)
    })
    history.sort(key=lambda x: x["episode"])
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    # 3. Generate GIF replay if completed_episode is a checkpoint episode
    if completed_episode in [100, 500, 1000, 5000]:
        gif_dir = "website/replays"
        os.makedirs(gif_dir, exist_ok=True)
        gif_path = os.path.join(gif_dir, f"checkpoint_{completed_episode}.gif")
        print(f"[GIF RECORDING] Generating replay GIF for Episode {completed_episode}...")
        try:
            generate_replay_gif(agent, env.state_mode, gif_path)
            print(f"[GIF RECORDING] Saved replay to {gif_path}")
        except Exception as e:
            print(f"[GIF RECORDING] Failed to generate GIF: {e}")


def train_headless(agent, env, target_episodes):
    print(f"\n[INFO] Starting Headless Training Mode (Target: {target_episodes} episodes)")
    print(f"Hyperparameters: alpha={agent.alpha}, gamma={agent.gamma}, decay={agent.decay_rate}, state={env.state_mode.upper()}")
    print("=" * 80)

    start_time = time.time()

    last_hits = env.hits
    last_misses = env.misses
    recent_rewards = []
    recent_hits = 0
    recent_misses = 0

    while env.episode <= target_episodes:
        state = env.get_state()
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)

        if done:
            agent.decay_epsilon()
            
            # Record episode stats
            ep_reward = env.episode_history[-1]
            ep_hits = env.hits - last_hits
            ep_misses = env.misses - last_misses
            last_hits = env.hits
            last_misses = env.misses
            
            recent_rewards.append(ep_reward)
            recent_hits += ep_hits
            recent_misses += ep_misses
            
            completed_episode = env.episode - 1
            
            # Checkpoint schedule
            if completed_episode in [100, 500, 1000, 5000]:
                ckpt_file = f"checkpoints/qtable_{completed_episode}.pkl"
                with open(ckpt_file, "wb") as f:
                    pickle.dump({
                        "version": "ternary_vy",
                        "q_table": agent.q_table,
                        "epsilon": agent.epsilon,
                        "episode": env.episode
                    }, f)
                print(f"[CHECKPOINT SAVED] File: {ckpt_file}")

            if completed_episode % 100 == 0:
                total_recent = recent_hits + recent_misses
                hit_rate = (recent_hits / total_recent * 100) if total_recent > 0 else 0.0
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                
                print(f"Episode: {completed_episode:<5} | Hit Rate: {hit_rate:>5.1f}% | "
                      f"Q-table Size: {len(agent.q_table):<5} | Epsilon: {agent.epsilon:.4f} | "
                      f"Average Reward: {avg_reward:>6.2f}")
                
                save_replays_and_stats(agent, env, completed_episode, avg_reward, hit_rate)
                
                recent_rewards = []
                recent_hits = 0
                recent_misses = 0

    elapsed = time.time() - start_time
    print(f"\n[INFO] Headless training completed in {elapsed:.1f}s.")


def train_visual(agent, env, target_episodes):
    # Initialize Pygame explicitly before font operations
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong RL Laboratory: Visual Training")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("Outfit", 18, bold=True)
        small_font = pygame.font.SysFont("Outfit", 14, bold=True)
        large_font = pygame.font.SysFont("Outfit", 22, bold=True)
    except Exception:
        font = pygame.font.Font(None, 22)
        small_font = pygame.font.Font(None, 16)
        large_font = pygame.font.Font(None, 26)

    running = True
    show_inspector = False
    show_heatmap = False
    show_educational = False
    start_time = time.time()

    last_hits = env.hits
    last_misses = env.misses
    recent_rewards = []
    recent_hits = 0
    recent_misses = 0

    while running and env.episode <= target_episodes:
        # Input Processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_i:
                    show_inspector = not show_inspector
                    show_heatmap = False; show_educational = False
                elif event.key == pygame.K_h:
                    show_heatmap = not show_heatmap
                    show_inspector = False; show_educational = False
                elif event.key == pygame.K_e:
                    show_educational = not show_educational
                    show_inspector = False; show_heatmap = False

        state = env.get_state()
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state, done)

        if done:
            agent.decay_epsilon()
            
            # Record episode stats
            ep_reward = env.episode_history[-1]
            ep_hits = env.hits - last_hits
            ep_misses = env.misses - last_misses
            last_hits = env.hits
            last_misses = env.misses
            
            recent_rewards.append(ep_reward)
            recent_hits += ep_hits
            recent_misses += ep_misses
            
            completed_episode = env.episode - 1
            
            # Checkpoint schedule
            if completed_episode in [100, 500, 1000, 5000]:
                ckpt_file = f"checkpoints/qtable_{completed_episode}.pkl"
                with open(ckpt_file, "wb") as f:
                    pickle.dump({
                        "version": "ternary_vy",
                        "q_table": agent.q_table,
                        "epsilon": agent.epsilon,
                        "episode": env.episode
                    }, f)
                print(f"[CHECKPOINT SAVED] File: {ckpt_file}")

            if completed_episode % 100 == 0:
                total_recent = recent_hits + recent_misses
                hit_rate = (recent_hits / total_recent * 100) if total_recent > 0 else 0.0
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                
                print(f"Episode: {completed_episode:<5} | Hit Rate: {hit_rate:>5.1f}% | "
                      f"Q-table Size: {len(agent.q_table):<5} | Epsilon: {agent.epsilon:.4f} | "
                      f"Average Reward: {avg_reward:>6.2f}")
                
                save_replays_and_stats(agent, env, completed_episode, avg_reward, hit_rate)
                
                recent_rewards = []
                recent_hits = 0
                recent_misses = 0

        # Render
        screen.fill(COLOR_BG)
        pygame.draw.rect(screen, COLOR_SIDEBAR_BG, (0, 0, SIDEBAR_WIDTH, HEIGHT))
        pygame.draw.line(screen, COLOR_WALLS, (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, HEIGHT), 3)

        arena_center_x = SIDEBAR_WIDTH + (WIDTH - SIDEBAR_WIDTH) // 2
        pygame.draw.line(screen, COLOR_WALLS, (arena_center_x, 0), (arena_center_x, HEIGHT), 2)

        env.ball.draw(screen)
        env.paddle.draw(screen)

        # Flash fx
        if env.flash_timer > 0:
            fs = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            fs.fill((*env.flash_color, int(env.flash_timer / env.flash_max * 80)))
            screen.blit(fs, (0, 0))
            env.flash_timer -= 1

        # Sidebar HUD Drawing
        screen.blit(large_font.render("PONG RL LABORATORY", True, COLOR_RED), (20, 15))
        screen.blit(small_font.render(f"State Mode: {env.state_mode.upper()}  |  VISUAL TRAIN", True, COLOR_CYAN), (20, 42))

        # Performance Dashboard Card
        c1 = pygame.Rect(20, 65, 280, 180)
        draw_card(screen, c1, "PERFORMANCE DASHBOARD", font)
        total_attempts = env.hits + env.misses
        hit_rate = (env.hits / total_attempts * 100) if total_attempts > 0 else 0.0
        elapsed = time.time() - start_time
        m, s = divmod(int(elapsed), 60)
        time_str = f"{m:02d}:{s:02d}"

        stats_lines = [
            f"Episode: {env.episode}",
            f"Hit Rate: {hit_rate:.1f}% (Best: {env.best_hit_rate:.1f}%)",
            f"Total Hits: {env.hits} | Misses: {env.misses}",
            f"Q-Table Size: {len(agent.q_table)} states",
            f"Training Time: {time_str}"
        ]
        for idx, line in enumerate(stats_lines):
            color = COLOR_YELLOW if "Hit Rate" in line else COLOR_TEXT_MAIN
            screen.blit(font.render(line, True, color), (35, 95 + idx * 25))

        # Decision Telemetry Card
        c2 = pygame.Rect(20, 255, 280, 140)
        draw_card(screen, c2, "DECISION TELEMETRY", font)
        screen.blit(small_font.render(f"STATE: {state}", True, COLOR_TEXT_STATE), (35, 280))
        
        vy_val = state[3] if len(state) == 5 else state[2]
        vx_val = state[4] if len(state) == 5 else state[3]
        vy_lbl = "UP" if vy_val == 0 else ("DOWN" if vy_val == 1 else "FLAT")
        vx_lbl = "LEFT" if vx_val == 0 else "RIGHT"
        
        screen.blit(font.render(f"Ball dY: {vy_lbl} | E: {agent.epsilon:.4f}", True, COLOR_TEXT_MAIN), (35, 305))
        screen.blit(font.render(f"ACTION: {Action.NAMES[action]} ({action})", True, COLOR_TEXT_MAIN), (35, 330))

        # Action arrow indicators
        indicator_x = 260
        indicator_y = 355
        if action == Action.UP:
            pygame.draw.polygon(screen, COLOR_TEXT_STATE, [(indicator_x, indicator_y - 12), (indicator_x - 12, indicator_y + 6), (indicator_x + 12, indicator_y + 6)])
        elif action == Action.DOWN:
            pygame.draw.polygon(screen, COLOR_TEXT_STATE, [(indicator_x, indicator_y + 12), (indicator_x - 12, indicator_y - 6), (indicator_x + 12, indicator_y - 6)])
        else:
            pygame.draw.rect(screen, COLOR_YELLOW, (indicator_x - 10, indicator_y - 10, 20, 20), border_radius=3)

        # Bellman & Q-Values Card
        c3 = pygame.Rect(20, 405, 280, 145)
        draw_card(screen, c3, "Q-VALUES & HYPERPARAMS", font)
        q_vals = agent.get_q_values(state)
        
        max_q = q_vals.max()
        min_q = q_vals.min()
        all_same = (max_q == min_q)
        for idx, act in enumerate(["STAY", "UP", "DOWN"]):
            val = q_vals[idx]
            col = COLOR_TEXT_MAIN
            if not all_same:
                if val == max_q: col = COLOR_TEXT_STATE
                elif val == min_q: col = COLOR_RED
            marker = " ◄" if idx == action else ""
            screen.blit(font.render(f"Q({act}): {val:+.3f}{marker}", True, col), (35, 435 + idx * 24))
        
        h_str = f"a={agent.alpha}  g={agent.gamma}  decay={agent.decay_rate}"
        screen.blit(small_font.render(h_str, True, COLOR_CYAN), (35, 518))

        # Mini Hit Rate Graph Card
        c4 = pygame.Rect(20, 560, 280, 90)
        draw_card(screen, c4, "LEARNING HISTORY", font)
        draw_mini_graph(screen, env.episode_history, pygame.Rect(30, 590, 260, 50), COLOR_TEXT_STATE, "", small_font)

        # Footer
        screen.blit(small_font.render("[I] Inspect | [H] Heatmap | [E] Guide | [ESC] Quit", True, COLOR_WALLS), (20, 665))

        # Overlay drawers
        if show_inspector:
            draw_inspector(screen, state, agent, font, large_font)
        elif show_heatmap:
            draw_heatmap(screen, agent, env.state_mode, font, large_font)
        elif show_educational:
            draw_educational(screen, font, large_font)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def train_mode(args):
    alpha = args.alpha
    gamma = args.gamma
    decay = args.epsilon_decay
    start_eps = args.epsilon_start
    target_episodes = args.episodes
    resume_path = args.resume

    if resume_path:
        if not os.path.exists(resume_path):
            print(f"[ERROR] Checkpoint not found: {resume_path}")
            sys.exit(1)
        with open(resume_path, "rb") as f:
            ckpt = pickle.load(f)
        
        is_compatible = False
        if isinstance(ckpt, dict):
            q_table = ckpt.get("q_table", {})
            epsilon = ckpt.get("epsilon", start_eps)
            start_episode = ckpt.get("episode", 1)
            if ckpt.get("version") == "ternary_vy":
                is_compatible = True
        else:
            q_table = ckpt
            epsilon = start_eps
            start_episode = 1
        
        # Infer state mode from Q-table keys
        if len(q_table) > 0:
            first_key = list(q_table.keys())[0]
            state_mode = "basic" if len(first_key) == 4 else "advanced"
        else:
            state_mode = args.state

        if not is_compatible:
            print("=" * 80)
            print(" [WARNING] COMPATIBILITY ALERT")
            print(" The resumed checkpoint does not match the new state representation (ternary vy).")
            print(" Continuing training may cause inconsistent updates and state space misalignment.")
            print("=" * 80)
            
        agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, decay_rate=decay)
        agent.q_table = q_table
        env = Environment(state_mode=state_mode)
        env.episode = start_episode
        print(f"Resuming training from file '{resume_path}' at episode {env.episode} in '{env.state_mode}' mode.")
    else:
        agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=start_eps, decay_rate=decay)
        env = Environment(state_mode=args.state)
        print(f"Starting fresh training in '{env.state_mode}' mode.")

    if args.render:
        train_visual(agent, env, target_episodes)
    else:
        train_headless(agent, env, target_episodes)


# ── Execution Entrypoint ──────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong RL Laboratory")
    
    # Modes
    parser.add_argument("--train", action="store_true", help="Start training the Q-learning agent")
    parser.add_argument("--render", action="store_true", help="Render training visually (requires --train)")
    parser.add_argument("--replay", type=str, default=None, metavar="CHECKPOINT", help="Demonstrate behavior from checkpoint file")
    parser.add_argument("--analyze", type=str, default=None, metavar="CHECKPOINT", help="Print offline analysis report of a checkpoint file")
    parser.add_argument("--resume", type=str, default=None, metavar="CHECKPOINT", help="Resume training from a checkpoint file")
    
    # Hyperparameters
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate (alpha)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor (gamma)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Exploration decay factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration probability")
    parser.add_argument("--episodes", type=int, default=5000, help="Target episode budget for training loops")
    parser.add_argument("--state", choices=["basic", "advanced"], default="advanced", help="Discretization complexity configuration")

    args = parser.parse_args()

    # Route CLI actions
    if args.analyze:
        analyze_checkpoint(args.analyze)
    elif args.replay:
        replay_mode(args.replay)
    elif args.train or args.resume:
        train_mode(args)
    else:
        parser.print_help()
        sys.exit(0)
