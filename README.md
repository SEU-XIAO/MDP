# Risk-Aware Path Planning with D3QN

This project trains an RL agent in a square grid map with circular risk zones. The agent must reach the goal while avoiding risk when possible, but it may enter risky areas when necessary.

## Features

- 3-channel global observation map with shape `(3, 64, 64)`:
  - Channel 0: fused risk probability map
  - Channel 1: goal potential field map
  - Channel 2: one-hot agent position map
- D3QN model (CNN + Dueling heads + Double DQN target update)
- Prioritized Experience Replay (PER)
- Heuristic exploration mixed with epsilon-greedy
- Environment randomization for better generalization

## Project Structure

- `configs/scenario.json`: map and enemy risk-zone configuration
- `src/environment/risk_grid_env.py`: gymnasium environment and reward logic
- `src/models/d3qn.py`: CNN + Dueling DQN network
- `src/replay/per_buffer.py`: prioritized replay buffer
- `src/agent/d3qn_agent.py`: action selection, learning, checkpoint I/O
- `train.py`: training entry
- `evaluate.py`: evaluation entry

## Reward Function

The per-step reward is:

`R = R_goal + R_step + R_risk + R_guide`

- `R_goal`: +100 at goal
- `R_step`: -1 per move
- `R_risk`: `-lambda * P^2`
- `R_guide`: `omega * (dist_prev - dist_curr)`

This encourages shortest safe paths and allows controlled risk-taking when detours are too costly.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --episodes 1200 --scenario configs/scenario.json
```

## Evaluate

```bash
python evaluate.py --model checkpoints/best_model.pt --episodes 20
```

## Visualize Policy Trajectory

Run an interactive real-time visualization to inspect how the trained agent moves.

```bash
python visualize_policy.py --model checkpoints/best_model.pt --episodes 1 --enemy-jitter 0 --start-jitter 0
```

Useful options:

- `--pause 0.08`: frame interval in seconds.
- `--epsilon 0.0`: keep greedy policy (no exploration).
- `--enemy-jitter 0 --start-jitter 0`: evaluate on static scenario.
- `--enemy-jitter 2 --start-jitter 2`: evaluate with random perturbations.

## Visual Scenario Designer (64x64)

Use the interactive GUI to design a fixed 64x64 scenario, drag circular risk areas, and export JSON.

```bash
python scenario_designer.py
```

Main interactions:

- Move Enemy mode: drag enemy centers on the grid.
- Set Start mode: click any cell to set start position.
- Set Goal mode: click any cell to set goal position.
- Add or remove enemies from the list.
- Edit each enemy's detection zones in JSON form, then apply.
- Save JSON to export a scenario file compatible with this project.

## Notes

- Action space uses 8-direction movement.
- Enemy positions and start position are randomized each episode by default.
- For faster debugging, run fewer episodes first, e.g. `--episodes 30`.
