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
| **sharpa-tacmap** | Tactile sensor integration in Isaac Lab | M2/M3: needed for higher-fidelity tactile sim |
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

| ID | Title | Status | Experiments |
|----|-------|--------|-------------|
| M1 | [Reproduce SHARPA RL results](M1_reproduce_sharpa.md) | ✅ done | EXP-000..011c |
| M2A | [Close distillation gap](M2_ablation.md) (Phase A) | ✅ done | EXP-012..014 |
| **M3** | [Generalize to RoboEra xhand](M3_extend.md) | **in progress** | EXP-015+ |
| M2B/C/D | [Component ablations + reward decomp + profiling](M2_ablation.md) | **deferred (between M3 and M4)** | — |
| M4 | [Sim2real & deployment on xhand](M4_sim2real.md) | planned | — |

## Execution order

1. ✅ M1 — Reproduce SHARPA on SharpaWave (both single-scale and multi-scale)
2. ✅ M2 Phase A — Close the distillation gap (found root cause: scale missing from priv_info)
3. **→ M3 — Port recipe to RoboEra xhand** (URDF + 12 DOF, grasp cache, retrain, distill, eval)
4. M2 Phase B/C/D — Component ablations + reward decomposition + profiling, now on BOTH SharpaWave and xhand → better informed conclusions
5. M4 — Sim2real deployment on real xhand, using insights from steps 3+4

**Why this order?** M2 B/C/D is about *understanding* the recipe so we can tune it on new hardware. It's most useful right before M4, and doing it after M3 means we can run each ablation on two hands instead of one — stronger claims about which components generalize.
