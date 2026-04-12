# SHARPA RL Lab

## Project Overview

Reinforcement learning sim2real pipeline for **in-hand object rotation** using the SharpaWave tactile robotic hand. Built on NVIDIA Isaac Lab for massively parallel physics simulation (16,384 envs) with PPO training and ProprioAdapt distillation for real-world deployment.

**Goal**: Learn and generalize the SHARPA RL library as a tool. Reproduce their published results first, then extend.

## Architecture

```
rl_isaaclab/
  algo/          # RL algorithms (PPO, ProprioAdapt distillation)
  scripts/       # Entry points: train.py, play.py, deploy.py, gen_grasp.py
  tasks/         # Isaac Lab task definitions (env + config)
  wrapper/       # Gym-style wrappers for Isaac Lab envs
  utils/         # Helpers (keyboard, events, misc)
assets/          # USD/URDF robot + object models, tactile sensor maps
cache/           # Grasp cache (.npy) - generated, not committed
pretrained/      # Shipped pretrained checkpoints
logs/            # Training outputs (TensorBoard + checkpoints) - gitignored
experiments/     # Autoresearch experiment log (committed)
```

## Key Commands

```bash
# Environment setup (use pixi, not conda)
pixi install

# Step 1: Generate grasp cache (must run before training)
pixi run gen-grasp

# Step 2: Train base policy (PPO, ~300M steps)
pixi run train

# Step 3: Visualize trained policy
pixi run play -- --load_path logs/.../stage1_nn/last.pth

# Step 4: Distillation (ProprioAdapt, stage 2)
pixi run train-distill -- --load_path logs/.../stage1_nn/last.pth

# Step 5: Play distilled policy
pixi run play-distill -- --load_path logs/.../stage2_nn/last.pth

# Raw commands (when pixi tasks aren't enough)
python rl_isaaclab/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --num_envs 16384
python rl_isaaclab/scripts/gen_grasp.py --task Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0 --headless
python rl_isaaclab/scripts/play.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 16 --load_path ${pth}
```

## Task Registration (Gym IDs)

- `Isaac-Inhand-Rotate-Sharpa-Wave-v0` — main training task
- `Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0` — grasp cache generation
- `Isaac-Inhand-Rotate-Deploy-Sharpa-Wave-v0` — real hardware deployment

## Training Pipeline

1. **Grasp cache** → stable initial hand poses (saved to `cache/`)
2. **PPO Stage 1** → train policy with privileged info + gravity curriculum
3. **ProprioAdapt Stage 2** → distill privileged info into proprioceptive history encoder
4. **Deploy** → run on real SharpaWave hand

## Key Hyperparameters (ppo_cfg.yaml)

- num_actors: 16384, horizon: 8, minibatch: 32768
- lr: 5e-3, gamma: 0.99, tau: 0.95, clip: 0.2
- mini_epochs: 5, max_steps: 300M
- save_frequency: 500 iterations

## Hardware Notes

- **GPU**: RTX 5090 (Blackwell, compute capability 12.0)
- **Known issues**: IRAY rendering broken, TiledCamera may hang, rendering artifacts on Ubuntu
- **Workarounds**: Use `--headless` mode, may need PyTorch 2.7+ nightly, avoid IRAY renderer
- **RAM**: 32GB minimum required

## Roadmap

See `roadmap/README.md` for milestones. Each experiment references a milestone via the `milestone` column in `experiments/experiments.tsv`.

- **M1**: Reproduce SHARPA results (current)
- **M2**: Understand & ablate components
- **M3**: Extend to new objects/tasks
- **M4**: Improve sim2real transfer

## Autoresearch Convention (Karpathy-style)

We follow an iterative experiment loop. Every experiment MUST be logged:

1. **Hypothesis** — what we expect and why
2. **Change** — what was modified (code, config, env)
3. **Run** — exact command, duration, hardware
4. **Result** — metrics (reward, val_bpb equivalent, success rate)
5. **Decision** — keep (commit) or revert (git reset)
6. **Next** — what to try next based on this result

All experiments are logged in `experiments/EXPERIMENTS.md` with sequential IDs.
Git commits for successful experiments reference the experiment ID.

### Experiment Metrics

- **Primary**: mean episode reward (from TensorBoard)
- **Secondary**: episode length, success rate (object not dropped), rotation speed
- **Infrastructure**: wall-clock time, GPU memory usage, num_envs achieved

## Code Style

- Python 3.10+, PyTorch
- Isaac Lab conventions (DirectRLEnv, ConfigClass dataclasses)
- No type annotations on unchanged code
- Keep diffs minimal and focused

## Dependencies

Managed via `pixi.toml` (Python 3.11 from conda-forge) + pip inside pixi env.

### Validated Install Sequence (RTX 5090)
```bash
pixi install                    # gets Python 3.11 + pip
pixi run install-deps           # or manually:
pixi run pip install 'isaaclab[isaacsim,all]==2.3.2.post1' --extra-index-url https://pypi.nvidia.com
pixi run pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pixi run pip install tensorboardX omegaconf
pixi run pip install -e .
```

**IMPORTANT**: Isaac Lab's pip install pulls PyTorch with cu126 by default. Must override with cu128 for RTX 5090 (sm_120).

Key versions:
- Isaac Lab 2.3.2.post1 + Isaac Sim 5.1.0.0
- PyTorch 2.7.0+cu128 (CUDA 12.8, sm_120 support)
- Python 3.11.15
- tensorboardX, gymnasium 1.2.0, omegaconf, numpy 1.26.0
