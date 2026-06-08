# Educational Pong Reinforcement Learning Laboratory

Welcome to the **Unified Pong Reinforcement Learning Laboratory** status page. This document outlines the mathematical foundation, structural layout, hyperparameter profiles, and expected learning progression of your Q-learning agent.

---

## 1. Mathematical Foundation (The Bellman Equation)

The agent learns to optimize its behavior using the **Temporal Difference (TD) Q-learning** update rule, which is a tabular representation of the Bellman Optimality Equation:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### Terms Definition:
- **$Q(s, a)$**: The current value of taking action $a$ in state $s$.
- **$\alpha$ (Learning Rate)**: Controls how much the new estimate overrides the old estimate ($0.1$).
- **$R(s, a)$**: The reward received after taking action $a$ in state $s$ ($+1$ for hit, $-10$ for miss).
- **$\gamma$ (Discount Factor)**: Determines the importance of future rewards ($0.95$).
- **$\max_{a'} Q(s', a')$**: The maximum estimated future reward starting from the next state $s'$.
- **$Q(s, a) - \dots$**: The Temporal Difference error ($R + \gamma \max Q' - Q$).

---

## 2. State Space Layout & Discretization

To make the environment learnable within a tabular representation, the continuous game screen ($1200 \times 700$ screen space, with an $880 \times 700$ playable game arena) is discretized into a finite state space:

$$\text{State Space Size} = \text{Ball X Zone} \times \text{Ball Y Zone} \times \text{Paddle Y Zone} \times \text{Ball } v_y \text{ State} \times \text{Ball } v_x \text{ State}$$

$$\text{Total States} = 10 \times 10 \times 10 \times 3 \times 2 = 6,000 \text{ states}$$

### State Representation Vector:
`(ball_x_zone, ball_y_zone, paddle_y_zone, ball_vy_direction, ball_vx_direction)`

1. **`ball_x_zone`** (0 to 9): Represents the horizontal coordinate of the ball relative to the playable arena width:
   $$\text{Zone} = \left\lfloor \frac{\text{Ball X} - \text{Sidebar Width}}{\text{Arena Width} / 10} \right\rfloor$$
2. **`ball_y_zone`** (0 to 9): Represents the vertical coordinate of the ball relative to the window height:
   $$\text{Zone} = \left\lfloor \frac{\text{Ball Y}}{\text{Window Height} / 10} \right\rfloor$$
3. **`paddle_y_zone`** (0 to 9): Represents the vertical position of the paddle center relative to the window height:
   $$\text{Zone} = \left\lfloor \frac{\text{Paddle Center}}{\text{Window Height} / 10} \right\rfloor$$
4. **`ball_vy_direction`** ($0$, $1$, or $2$): The vertical velocity state ($0$ for up $v_y < 0$, $1$ for down $v_y > 0$, $2$ for flat $v_y = 0$).
5. **`ball_vx_direction`** ($0$ or $1$): The horizontal velocity state ($0$ for left $v_x < 0$, $1$ for right $v_x \ge 0$).

---

## 3. Discretization Detail Overview

By including the `ball_x_zone` and `ball_vx_direction` into the state space, we have resolved the **Partial Observability** issue (previously the agent was blind to horizontal movement). Now, the agent can:
- Anticipate the arrival time of the ball based on its distance (`ball_x_zone`).
- Detect whether the ball is moving away from or toward it (`ball_vx_direction`).
- Track the descent or ascent speed of the ball (`ball_vy_direction`).

---

## 4. Hyperparameter Profiles

The default hyperparameters for the training run are:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate ($\alpha$)** | `0.1` | Step size for updates. |
| **Discount Factor ($\gamma$)** | `0.95` | Focus on immediate and short-term future returns. |
| **Epsilon Start ($\epsilon_0$)** | `1.0` | Initial exploration rate (100% random actions). |
| **Epsilon Decay Rate** | `0.995` | Exploration factor decays by 0.5% per finished episode. |
| **Minimum Epsilon ($\epsilon_{\text{min}}$)** | `0.05` | Constant background exploration rate. |

---

## 5. Expected Learning Outcomes

Below are the learning progress metrics you will observe when training the model:

### Milestone 1: Episode 100
- **Epsilon**: $\approx 0.60$
- **Hit Rate**: $10\% - 20\%$
- **Q-table Size**: $\approx 1,300 - 1,800$ states
- **Behavior**: The agent moves semi-randomly but is beginning to cluster around the ball's trajectory when it is close.

### Milestone 2: Episode 500
- **Epsilon**: $\approx 0.08$
- **Hit Rate**: $45\% - 60\%$
- **Q-table Size**: $\approx 3,000 - 3,600$ states
- **Behavior**: The agent actively tracks the ball's Y-coordinate, adjusting its positioning as the ball travels across the screen.

### Milestone 3: Episode 1000
- **Epsilon**: `0.05` (saturates at minimum threshold)
- **Hit Rate**: $70\% - 85\%$
- **Q-table Size**: $\approx 3,600 - 3,900$ states
- **Behavior**: The agent displays sharp, anticipation-driven tracking. It moves to intercept the ball with high precision and minimizes unnecessary movement when the ball is on the other side of the arena.

---

## 6. CLI Usage Guide

Run these commands from the root directory (`pingpongballRL/`):

### 1. Headless Training (Fastest, Default)
```bash
python pong_rl_lab.py --train --episodes 1000
```

### 2. Visual Training (Rendered)
```bash
python pong_rl_lab.py --train --render --episodes 1000
```

### 3. Resume Training from Checkpoint
```bash
python pong_rl_lab.py --resume checkpoints/qtable_500.pkl --episodes 1000
```

### 4. Replay Mode (Demonstrate Checkpoint)
```bash
python pong_rl_lab.py --replay checkpoints/qtable_500.pkl
```

### 5. Analyze Mode (Inspect Checkpoint Metrics)
```bash
python pong_rl_lab.py --analyze checkpoints/qtable_500.pkl
```
