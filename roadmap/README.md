# Roadmap

Each milestone gets its own file. Experiments in `experiments/experiments.tsv` reference milestones via the `milestone` column.

## Context: Sharpa Ecosystem

**Sharpa Robotics** (Singapore/Shanghai/Mountain View) builds the **SharpaWave** — a 22-DOF dexterous hand with 1000+ tactile pixels per fingertip, sub-mm resolution, 0.005N–20N+ force range. Mass production since Dec 2025. Also building the **Sharpa North** humanoid (63 DOF, 0.02s reaction time).

**Key AI models**: CraftNet (hierarchical VTLA, ~100Hz tactile control), MoDE-VLA (mixture of dexterous experts, 34% success on contact-rich tasks like apple peeling/gear assembly).

**This repo** (`sharpa-rl-lab`) is their open-source RL sim2real demo for in-hand cylinder rotation on the SharpaWave hand, built on Isaac Lab.

### Sharpa GitHub Org ([github.com/sharpa-robotics](https://github.com/sharpa-robotics))

| Repo | Purpose | Relation to this work |
|------|---------|----------------------|
| **sharpa-rl-lab** (this repo) | RL sim2real demo for in-hand rotation | — |
| **sharpa-tacmap** | Tactile sensor integration in Isaac Lab | M2/M4: higher-fidelity tactile sim |
| **sharpa-urdf-usd-xml** | SharpaWave hardware asset files | Already vendored in `assets/SharpaWave/` |
| **sharpa-tactile-sensor-assets** | Static tactile sensor component files | Already vendored in `assets/tactile_ha4_map/` |
| **sharpa-manus-sdk** | Manus MetaGloves Pro teleop SDK | M4: useful for collecting demo data |
| **sharpa-opencad-library** | Hardware CAD design | Not directly relevant |
| **sharpa-community** | Discussion forum | Reference |

### Other Sharpa Tools
- **SharpaPilot**: Control application, compatible with Isaac Lab/PyBullet/MuJoCo/Isaac Gym
- **Tacmap**: NVIDIA-partnered high-fidelity tactile simulation framework — this repo uses a simpler tactile model via `assets/tactile_ha4_map/*.npy`, upgrading to tacmap is a future direction

### Related Academic Work
- **AnyRotate** (2024): Gravity-invariant in-hand rotation with sim-to-real touch
- **RotateIt** (Qi et al. 2023): General in-hand rotation with vision + touch + proprioception
- **Touch Dexterity**: In-hand rotation using only tactile sensing
- **NeuralFeels**: Visuotactile perception for in-hand manipulation

---

## Milestones

Two parallel tracks:
- **Track A** — SharpaWave (the original SHARPA RL target; hardware on the way)
- **Track B** — RoboEra xhand (we have URDFs + SDK locally; smaller 12-DOF hand for generalization)

| ID | Track | Title | Status | Experiments |
|----|-------|-------|--------|-------------|
| M1 | A | [Reproduce SHARPA RL results (SharpaWave sim)](M1_reproduce_sharpa.md) | ✅ done | EXP-000..011c |
| M2A | A | [Close distillation gap](M2_ablation.md) (Phase A) | ✅ done | EXP-012..014 |
| **M3** | B | [Port RL recipe to xhand (sim)](M3_xhand_sim.md) | **in progress** | EXP-015+ |
| M2B/C/D | A+B | [Component ablations + reward decomp + profiling](M2_ablation.md) | deferred until after M3 | — |
| M4 | A | [SharpaWave sim2real deployment](M4_sharpa_sim2real.md) | blocked on hardware arrival | — |
| M5 | B | [xhand sim2real deployment](M5_xhand_sim2real.md) | blocked on M3 completion | — |

### Track details

**Track A (SharpaWave)** — the canonical reproduction:
1. ✅ **M1**: Reproduce SHARPA's published single-scale + multi-scale results in sim
2. ✅ **M2A**: Fix the multi-scale distillation gap (root cause: scale missing from priv_info)
3. 🟡 **M4** (blocked): Deploy our trained policies on physical SharpaWave once hardware arrives

**Track B (RoboEra xhand)** — generalization proof + second hardware target:
1. 🟡 **M3** (active): Port the full recipe to xhand in sim — asset conversion, HandSpec refactor, grasp cache, PPO, distill, eval. Prove the recipe is hand-agnostic.
2. 🟡 **M5** (blocked on M3): Deploy xhand-trained policies on physical xhand (local hardware available)

**Cross-track (M2B/C/D)**: Component ablations + reward decomposition + profiling. Deferred until after M3 so we can run ablations on **both hands** for stronger conclusions. These inform both M4 and M5.

## Execution order

1. ✅ M1 — Reproduce SHARPA on SharpaWave (both single-scale and multi-scale)
2. ✅ M2A — Close the distillation gap
3. **→ M3 — Port recipe to xhand in simulation** (in progress: URDF conversion, HandSpec refactor, grasp cache, training)
4. M2 Phase B/C/D — Component ablations on both SharpaWave AND xhand baselines
5. M4 + M5 — sim2real deployments in parallel once respective hardware is ready

**Why M3 first vs M4**: SharpaWave hardware hasn't arrived yet. xhand hardware is available locally. Running M3 in sim on xhand exercises the whole pipeline under a new hand, proves the recipe is portable, and provides the second baseline for M2B/C/D ablations. When SharpaWave arrives, M4 can proceed in parallel with M5.
