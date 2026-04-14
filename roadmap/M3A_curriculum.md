# M3A: Curriculum + helper-force thread

**Branch**: `m3-curriculum`
**Parent**: `main @ 2997d61`
**Started**: 2026-04-14

## Why this thread exists

After 7 PPO variants on xhand with the side-pinch pose (see `roadmap/M3_xhand_sim.md`), every configuration plateaus at ~-1000 mean reward. Root cause: the cached "grasps" are transient contact states, so on reset the sphere/cylinder drops before PPO can react, and the agent's best strategy is "drop fast → minimize penalty before height_reset terminates the episode". PPO never sees the rotation reward.

The curriculum thread attempts to give PPO a softer gradient by making the early task MUCH easier, then progressively hardening it.

## Hypothesis

If we:
1. Remove or reduce the dropping penalty source (low gravity / external helper force)
2. Add a clear positive "alive" signal that rewards keeping the sphere near the palm
3. Only introduce rotation reward AFTER the agent can reliably hold

then PPO will first learn a stable hold policy, and rotation will follow naturally as a perturbation of the hold policy.

## Scope note (2026-04-14 clarification)

Curriculum lives at two levels, both valid and orthogonal:

1. **In-training curriculum** (this thread's current work): within a single
   training run, gradually change the env difficulty — gravity, penalties,
   object mass, etc. — via step-scheduled interpolation. Single PPO checkpoint,
   no stage handoff. Cheap and directly addresses the -1000 plateau caused by
   reset-on-drop. The `gravity_schedule` field in `xhand_env_cfg.py` (read by
   `sharpa_wave_env._get_dones`) implements this.

2. **Task-content curriculum** (future elevation): SEPARATE training runs for
   qualitatively different sub-goals ("hold only", "hold+rotate", "rotate
   different objects"), with a VLM-driven planner deciding what sub-goal to
   attempt next based on current policy performance. Each sub-goal produces a
   checkpoint that feeds the next. This is a level ABOVE the in-training
   curriculum and is an ELEVATION, not a replacement — it builds on top.
   Deferred until the in-training curriculum either succeeds or proves
   insufficient.

This document is scoped to level 1.

## Plan

### Step 1 — Verify gravity state (fastest check)

The `gravity_curriculum` flag in SharpaWave starts gravity at `-0.05 m/s²` (effectively zero) and only ramps up when `height_reset` rates fall below `5e-4`. For xhand with its unstable cache, reset rates stay high, so **gravity should already be near zero**.

**Action**: instrument one training run and dump `physics_sim_view.get_gravity()` every 200 steps. If gravity is genuinely near zero and PPO still can't learn, the problem is NOT gravity — it's reward shaping.

### Step 2 — Helper "anti-gravity" force

Add a per-step external force to the sphere equal to `mass * 9.81 * alpha`, where `alpha` is a curriculum knob:
- Training steps 0 – 20M: `alpha = 1.0` (fully cancels gravity, sphere floats in place)
- 20M – 60M: linear ramp `alpha = 1.0 → 0.0`
- 60M+: real gravity, no helper

This lets PPO learn finger placement first without fighting falls, then progressively learn to hold against real weight.

**Implementation**: `sharpa_wave_env.py` already applies `self.rb_forces` each sim step via `set_external_force_and_torque`. Add a per-env `helper_force` that is `+mass*g*alpha` in the -gravity direction, decaying on a schedule. Make it an opt-in cfg field `helper_force_alpha_schedule`.

### Step 3 — Clean up reward weights for Stage-1 hold

During the helper-force phase, disable rotation reward entirely and instead give:
- `+1.0 * is_inside_palm` per step (binary alive bonus — sphere within 10cm of palm centroid)
- Tiny contact reward (`+0.1 * num_fingers_touching_sphere`)
- Zero `work_penalty`, minimal `torque_penalty`

### Step 4 — Rotation-reward transition

Once the hold is stable (alive bonus > 300/episode consistently), flip rotation reward back on with full scale and turn off the alive bonus.

### Step 5 — Distill (if Stage 1 works)

Standard ProprioAdapt stage-2 using the curriculum stage-1 checkpoint.

## Risks / knowns

- **Risk 1**: Even with zero gravity and alive bonus, PPO may not find a policy that improves over "do nothing". The reset noise on joint positions might already put the hand far from contact. Mitigation: initialize from cached poses as before, noise down to 0.05 rad.
- **Risk 2**: The transition from helper-force to real gravity is where things often break. Mitigation: very slow linear ramp (20M steps) and keep the alive bonus during the ramp as a safety net.
- **Risk 3**: The gravity curriculum was DESIGNED for SharpaWave which can form stable grasps. Maybe xhand fundamentally can't, and no amount of curriculum will fix it. If so, fall back to Thread B (retargeting).

## Experiment tags

All experiments on this branch use `EXP-M3A-{num}`:
- `EXP-M3A-1`: gravity verification (diagnostic only, no training)
- `EXP-M3A-2`: helper_force at alpha=1.0 constant (no decay), no rotation reward, alive bonus only — can PPO learn to hold a floating sphere?
- `EXP-M3A-3`: helper_force with linear decay
- `EXP-M3A-4`: rotation reward transition
- `EXP-M3A-5`: full pipeline + stage-2 distillation

## Success criteria

**Thread A succeeds if**: mean reward > +50 on stage-1 (the alive bonus dominates and agent holds reliably) AND stage-2 eval reaches > +100 (non-trivial rotation on xhand).

**Thread A fails if**: by `EXP-M3A-3` (the decay phase), reward drops back to ~-1000 plateau and doesn't recover. At that point, merge/discard and move to Thread B.
