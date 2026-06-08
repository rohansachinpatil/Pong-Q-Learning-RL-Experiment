# Pong RL Lab Dashboard: Deployment Guide

This guide explains how to run, test, and deploy the browser-based Reinforcement Learning dashboard for the Pong Q-learning agent.

---

## Architecture Overview

The system consists of two primary parts:

1. **Training Engine (`pong_rl_lab.py`)**: Runs the Q-learning agent. Every 100 episodes, it exports updated data to `website/data/stats.json` and `website/data/history.json`. At milestone checkpoints (100, 500, 1000, 5000 episodes), it also saves weights and renders animated GIF replays to `website/replays/checkpoint_[N].gif`.
2. **Telemetry Dashboard (`website/`)**: A responsive HTML5/CSS3 dashboard with a vanilla JavaScript telemetry pipeline and dynamic charting via Chart.js. It periodically fetches the exported stats and lazily unlocks/renders replays when checkpoint GIFs become available.

---

## Local Development & Testing

### 1. Generate Training Data & Replays
First, initialize and start the training process. The dashboard needs the generated JSON logs to show charts:

```bash
# Run headless training (fastest, ideal for generating data)
python pong_rl_lab.py --train

# OR run visual training (renders the Pygame environment in real time)
python pong_rl_lab.py --train --render
```

As training completes episodes, it will create/update:
* `website/data/stats.json` (Real-time telemetry and metadata)
* `website/data/history.json` (Historical logs for Chart.js)
* `website/replays/checkpoint_100.gif` (Saves when Episode 100 ends)
* `website/replays/checkpoint_500.gif` (Saves when Episode 500 ends)
* `website/replays/checkpoint_1000.gif` (Saves when Episode 1000 ends)
* `website/replays/checkpoint_5000.gif` (Saves when Episode 5000 ends)

### 2. Launch Local Server
Because the script uses `fetch()` requests to load the local JSON data files, browsers may block them if you try to open `website/index.html` directly (due to CORS policy rules for `file://` URIs). 

To view the dashboard, run a simple local web server from the project directory:

**Option A (Python - Built-in)**:
```bash
python -m http.server 8000
```
Then open [http://localhost:8000](http://localhost:8000) (or [http://localhost:8000/website](http://localhost:8000/website) depending on where you run the command).

**Option B (Node.js/npm)**:
```bash
npx live-server --port=8000
```

---

## Vercel Deployment

We have configured a `vercel.json` file in the project root to enable seamless, configuration-free deployment.

### Method 1: Vercel GitHub Integration (Recommended)
1. Push your repository to GitHub, GitLab, or Bitbucket.
2. Go to the [Vercel Dashboard](https://vercel.com/dashboard) and click **Add New > Project**.
3. Select your repository.
4. Keep the default settings (Vercel will automatically read the `vercel.json` and rewrite root requests to the `website/` directory).
5. Click **Deploy**.

### Method 2: Vercel CLI (Command Line)
If you prefer deploying directly from your local terminal:

1. Install the Vercel CLI:
   ```bash
   npm install -g vercel
   ```
2. Log in to Vercel:
   ```bash
   vercel login
   ```
3. Run the deployment command from the project root:
   ```bash
   vercel
   ```
4. For production deployment, promote the build:
   ```bash
   vercel --prod
   ```

---

## Telemetry Metrics Guide

The dashboard monitors the following values:

* **Episode Progress**: Visualizes how close the agent is to completing its 5,000 episode training budget.
* **Hit Rate**: The percentage of incoming balls successfully returned by the paddle. Reaches ~97% for the expert agent.
* **Epsilon**: The exploration factor. Starts at `1.0` (fully random actions) and decays exponentially to `0.05` (mostly exploitation of learned Q-values).
* **Q-table States**: The number of unique game states the agent has encountered and indexed. The ternary $v_y$ state representation has a maximum space of 6,000 states.
* **Average Reward**: The rolling mean reward calculated over the last 100 training episodes. Bounded in $[-5.0, +5.0]$ range.
