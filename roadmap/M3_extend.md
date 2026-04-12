# M3: Generalize — New Tasks & New Hands

**Goal**: Prove the SHARPA RL recipe (PPO + ProprioAdapt + Isaac Lab) works as a **reusable tool**, not just a SharpaWave-only demo. Generalize across tasks *and* embodiments.

**Depends on**: M1 (working baseline), M2 (know which components are load-bearing)

**Core thesis**: A dex-manipulation training recipe is hand-agnostic at the algorithm level but hand-specific at the asset / obs-shape / grasp-cache level. M3 verifies this and produces a templated path to swap hands or tasks.

## Part A — What's hand-specific vs hand-agnostic in this codebase

From reading the codebase, here's the decomposition:

### Hand-agnostic (should work unchanged for any hand)
- **PPO algorithm** (`rl_isaaclab/algo/ppo/ppo.py`)
- **ProprioAdapt distillation** (`rl_isaaclab/algo/padapt/padapt.py`)
- **ActorCritic model** (`rl_isaaclab/algo/models/models.py`) — auto-sizes from obs/action dims
- **Training/eval/play scripts** (`rl_isaaclab/scripts/`)
- **Reward structure** — rotate_reward is Z-axis angular velocity, works for any rotating-object task
- **Gym wrapper** (`rl_isaaclab/wrapper/`)

### Hand-specific (must change per hand)
| Component | File | Current SharpaWave value |
|-----------|------|-------------------------|
| URDF/USD asset | `assets/SharpaWave/right_sharpa_wave.usda` | 22-DOF hand model |
| `action_space` | `sharpa_wave_env_cfg.py:42` | 22 |
| `observation_space` | `sharpa_wave_env_cfg.py:43` | 192 |
| `action_scale` | `sharpa_wave_env_cfg.py:52` | 1/24 (hand inertia-dependent) |
| Fingertip body names | `sharpa_wave_env_cfg.py:224-229` | `right_{thumb,index,middle,ring,pinky}_fingertip` |
| Contact sensor prims | `sharpa_wave_env_cfg.py:135-200` | 10 sensors (5 elastomer + 5 DP) per finger |
| Initial hand pose | `sharpa_wave_env_cfg.py:69` | Pre-grasp orientation for SharpaWave |
| Grasp cache | `cache/*.npy` | **Must regenerate per hand** (~20 min) |
| Reward scales | `rotate_reward_scale` etc | Tuned for SharpaWave torque range |
| Elastomer material IDs | `sharpa_wave_env.py:139` | `[19, 20, 22, 24, 25]` (SharpaWave link indices) |

**Observation dim breakdown** (192 = `obs_buf_lag_history[-3:]` flattened):
- 22 joint positions
- 22 joint target offsets
- 5 contact forces (binary or analog)
- 15 contact positions (5 × 3D)
= 64 dims per frame × 3 frames = **192**

For a 12-DOF hand like xhand: obs becomes `(12 + 12 + 5 + 15) × 3 = 132` (assuming same 5-finger layout).

## Part B — Task generalization (easier)

Reuse the SharpaWave hand but change the task.

| Task | What to change | Complexity |
|------|---------------|-----------|
| **Multi-axis rotation** (X/Y in addition to Z) | Change `rot_axis` to random unit vector per env; update reward | Low — 1 day |
| **Target orientation reach** | New reward: `-angle_to_target`; new termination condition | Low — 1 day |
| **Sphere instead of cylinder** | Swap `assets/cylinder/cylinder.usd` for a sphere USD; regenerate grasp cache | Low — 0.5 day |
| **Cube** | New USD, new grasp cache, harder because corners | Medium — 2 days |
| **Irregular objects** | Multiple USDs, per-object reward scales | Medium — 3 days |
| **Contact-rich task** (like Sharpa's MoDE-VLA apple peeling) | New env class, new reward, new objects | Hard — week+ |

**Proposed M3-A experiment sequence**:
- M3-A1: Cylinder rotation around random axis (not just Z)
- M3-A2: Sphere rotation (isotropic object, easier)
- M3-A3: Cube rotation (hardest single-object task)

## Part C — Hand generalization (the big test)

### Target hand: RoboEra xhand
- **DOF**: 12 (vs SharpaWave's 22) — nearly half
- **Fingers**: 5 (thumb 3 DOF, index 3 DOF, middle/ring/pinky 2 DOF each)
- **Tactile**: 120-300 three-axis sensors per fingertip
- **Grip force**: 80N peak; 5kg single finger, 24kg full hand
- **Availability**: Product exists; [roboterax GitHub](https://github.com/roboterax) has humanoid URDF but **no public xhand URDF/USD yet**

### xhand path
1. **Blocker**: Need xhand URDF/USD. Either:
   - Contact RoboEra for URDF
   - Build URDF from datasheet dimensions
   - Use any public 12-DOF dex hand as a stand-in (e.g., Shadow Hand, Allegro) first

2. **Realistic first target**: **Allegro Hand** (16 DOF, 4 fingers, widely available in Isaac Lab) — proves the pipeline is portable, unblocks xhand later

### M3-B experiment sequence (Allegro as proof-of-concept)

| # | Experiment | Description | Est. effort |
|---|------------|-------------|-------------|
| M3-B1 | Port env to Allegro | Swap URDF, update action_space=16, update contact sensors (4 fingers not 5), regenerate grasp cache | 1-2 days |
| M3-B2 | Train Allegro cylinder rotation | 300M steps, measure convergence | ~1.5h compute |
| M3-B3 | Distill Allegro stage-2 | Apply M2 learnings on distillation | ~0.5-5h compute |
| M3-B4 | Eval vs published Allegro benchmarks | Compare against [Bi-DexHands](https://pku-marl.github.io/DexterousHands/) or [AnyRotate](https://arxiv.org/abs/2405.07391) numbers | 1 day |

Success = Allegro policy rotates a cylinder reliably without significantly re-tuning reward constants. This proves the recipe is hand-agnostic at the algorithmic level.

### Templating for future hands (deliverable of M3-B)

Factor out hand-specific config into a single `HandSpec` class:

```python
@configclass
class HandSpec:
    name: str
    usd_path: str
    action_space: int
    dof_names: list[str]
    fingertip_body_names: list[str]
    contact_sensor_bodies: list[str]  # per-finger elastomer links
    initial_pose: tuple
    action_scale: float
    elastomer_material_ids: list[int]

# Then:
SHARPA_WAVE = HandSpec(name="sharpa_wave", action_space=22, ...)
ALLEGRO = HandSpec(name="allegro", action_space=16, ...)
XHAND = HandSpec(name="xhand", action_space=12, ...)
```

Current code has these constants scattered across `sharpa_wave_env_cfg.py` and `sharpa_wave_env.py`. Consolidating them into a HandSpec is the main refactor M3 needs.

### M3-C: xhand once URDF is available

Direct port once we have the asset. Same sequence as M3-B:
1. Build `XHAND = HandSpec(...)`
2. Generate grasp cache
3. Train PPO
4. Distill
5. Evaluate

## Research backing

- **CrossDex** (ICLR 2025, [arxiv](https://arxiv.org/abs/2410.02479)): cross-embodiment policies via eigengrasp retargeting — fingertip-position obs + retargeted actions transfer zero-shot
- **Bi-DexHands** (PKU-MARL): Isaac Gym benchmark showing observation space decoupling works for policy transfer
- **ManipTrans** ([arxiv](https://arxiv.org/abs/2503.21860)): residual RL accelerates transfer when swapping morphologies
- **AnyRotate** (gravity-invariant in-hand rotation with tactile sim2real) — closest published analog to what we're reproducing

## Relation to Sharpa's broader work

- **CraftNet** System 0: low-level dexterous policies like ours are the bottom of their hierarchy. Porting to new hands lets their high-level planners target those hands.
- **MoDE-VLA**: mixture of experts for contact-rich tasks — M3-A3 (cube) + M3-A4 (contact-rich) builds toward this kind of capability
- **sharpa-urdf-usd-xml**: may have left-hand variant we haven't used yet — bimanual is another M3 extension
- **Tacmap** (Sharpa+NVIDIA): upgrading tactile sim fidelity is complementary to M3 — combining M3 (new hand) with tacmap (better tactile) is the realistic path to a general multi-hand pipeline

## Deliverables

- Factored `HandSpec` + one non-SharpaWave hand working end-to-end (Allegro)
- Reward-curve plot comparing SharpaWave vs Allegro on cylinder rotation
- Documented porting guide (`docs/porting_new_hand.md` or in this roadmap)
- xhand integration if URDF becomes available
