# Branch `m3-curriculum` — summary, conclusions, reproduction

_Last updated: 2026-05-31 (picked back up on a fresh laptop, RTX 5090 Laptop 24 GB)._

This doc answers: **(1) what was done on this branch vs the original SHARPA repo,
(2) the conclusions, (3) how to reproduce both the SHARPA results and ours.**

---

## 1. What this branch added on top of the original SHARPA repo

The upstream repo ships a SharpaWave (22-DOF) in-hand rotation recipe: grasp-cache
→ PPO (stage 1) → ProprioAdapt distillation (stage 2) → deploy. This branch did
three things:

### A. Reproduced and *exceeded* the SHARPA baseline (M1 + M2)
- **M1 single-scale**: stage-2 eval **1137.66** vs shipped pretrained 1040.63 (**+9.3 %**).
- **M1 multi-scale**: stage-2 eval **1031.46** vs pretrained 950.38 (**+8.5 %**).
- **M2A — closed the multi-scale distillation gap** with a *one-line* fix:
  `priv_info` did not contain object scale, so the distillation target was
  scale-invariant while the proprio input was scale-dependent → MSE had no
  consistent target. Fix: `self.priv_info_buf[:, 8] = self.env_scales`
  (`priv_info_dim` 8→9). Stage-1 reward jumped +17 %, stage-2 went from −7.6 %
  to +8.5 % vs pretrained. (EXP-012/013/014.)

### B. Made the recipe hand-agnostic and ported it to the RoboEra **xhand** (M3)
- URDF→USD conversion (`convert_xhand_urdf.py`, `merge_fixed_joints=False`).
- Hand-agnostic refactor: hardcoded `22/29/10` → `num_hand_dofs` / `num_sensors`,
  added `contact_sensor_body_names`, `num_hand_materials`, `elastomer_material_ids`,
  `fingertip_body_names`, per-hand `hand_init_pose`. New configs
  `xhand_env_cfg.py` + `xhand_grasp_env_cfg.py`, tasks `Isaac-Inhand-Rotate-Xhand-v0`
  and `Isaac-Inhand-Rotate-Grasp-Xhand-v0`.
- **VLM visual-diagnostics harness** (`diagnostics/vlm.py`, `scripts/visual_check.py`):
  renders the hand+object to a PNG and (optionally) asks Gemini "is this a
  trainable grasp?". Used to iterate the xhand pre-grasp pose ~30 times.
- Grasp tooling: incremental cond-true cache save + streak filter (so xhand's
  intermittent contacts can fill a cache without surviving a full 400-step
  gravity cycle), `synth_xhand_cache.py`, a standalone `eval.py`.

### C. Curriculum thread (M3A) — first positive xhand reward
- Step-scheduled **gravity curriculum** inside a single training run
  (`gravity_schedule`, read in `sharpa_wave_env._get_dones`): hold at g=0, then
  ramp to −9.81 over ~190 M agent steps.
- Opt-in **alive bonus** and **contact reward** terms in `_get_rewards`.

---

## 2. Conclusions

1. **The SHARPA recipe is sound and reproducible** — we beat the published
   numbers on both single- and multi-scale (M1) and fixed the one real bug in
   their multi-scale distillation (M2A). This part is *done and solid*.

2. **The recipe is hand-agnostic at the algorithm level but NOT at the
   grasp/asset level.** PPO, ProprioAdapt, the model, the reward shape all
   ported to xhand unchanged. What did *not* port for free is the **grasp**:
   SharpaWave forms a real opposable cage; xhand's near-coplanar fingers + one
   thumb do not.

3. **The xhand wall was the grasp, not the RL.** Every xhand PPO run with the
   flat palm-up pose plateaued at ~−1000 because the cached "grasps" are
   momentary contacts, not stable holds → the object drops on reset →
   `height_reset` ends the episode → PPO's optimum is "drop fast, minimize
   penalty". Two things broke the plateau (M3A):
   - **Root cause of the floor**: `work_penalty` raw ≈ 6000/step, so even a
     −0.02 weight gave −120/step and dominated → optimal policy = "don't move".
     Zeroing `work_penalty`/`torque_penalty` during the curriculum was the
     single biggest lever (−400 → +116 immediately).
   - **Gravity curriculum + alive bonus** let PPO learn a zero-g hold first,
     then keep it as gravity ramps in.
   - Result: **first positive xhand run, +128 peak at zero-g, decaying to ~+54
     steady-state under full gravity** (EXP-M3A-8, 250 M steps).

4. **+54 is a weak hold, not real rotation.** The positive reward is dominated
   by the alive bonus, not `rotate_reward`. The flat palm-up cradle is
   fundamentally marginal — gravity is *perpendicular* to the palm and the
   object is one nudge from rolling out the thumb-opposite gap. We were
   fighting the geometry with reward shaping. **This is the thread to pull next
   (see §4 and `roadmap/M3B_angled_palm.md`).**

5. **State of the machine (2026-05-31):** this is a *fresh laptop*. `logs/` is
   empty — the M3A stage-1 checkpoints (`logs/debug/2026-04-14_12-29-23/`) were
   on the old desktop and are **not here**. What we *do* have locally: the
   shipped SharpaWave pretrained checkpoints (`pretrained/`), all grasp caches
   (`cache/`), and all code/experiment logs. So SharpaWave results are
   immediately reproducible; xhand results require a retrain.

---

## 3. How to reproduce

### 3.0 One-time environment setup (this laptop)
```bash
curl -fsSL https://pixi.sh/install.sh | bash    # installs pixi to ~/.pixi/bin
export PATH="$HOME/.pixi/bin:$PATH"              # (add to ~/.bashrc)
cd ~/sharpa-rl-lab
pixi install                                     # Python 3.11 env
pixi run install-deps                            # isaaclab 2.3.2 + torch cu128 (~10 GB, slow)
```
Note: 24 GB laptop VRAM < the 32 GB desktop. Use `--num_envs 4096` (or 8192) for
xhand and `--num_envs 8192` for SharpaWave if 16384 OOMs.

### 3.1 SHARPA results (immediate — pretrained checkpoints + caches are present)

**Watch the shipped policy rotate (single-scale):**
```bash
pixi run play -- --num_envs 16 --load_path pretrained/0.5-0.5-1.pth \
  --cache cache/sharpa_grasp_linspace_0.5-0.5-1.npy
```

**Numeric eval of the shipped multi-scale pretrained (baseline 950.38):**
```bash
pixi run python rl_isaaclab/scripts/eval.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --num_envs 256 --num_episodes 256 \
  --algorithm ProprioAdapt --load_path pretrained/0.4-0.6-8.pth \
  --cache cache/sharpa_grasp_linspace_0.4-0.6-8.npy \
  --output_json experiments/repro_pretrained_multiscale.json
```

**Reproduce our M1/M2 win from scratch (≈94 min stage-1 + ≈31 min distill):**
```bash
# stage 1 (PPO, scale-aware priv_info already in code)
pixi run python rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --num_envs 16384
# stage 2 (distill) — point load_path at the stage1_nn/last.pth just produced
pixi run python rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --headless --num_envs 16384 \
  --algorithm ProprioAdapt --max_agent_steps 100000000 \
  --load_path logs/debug/<TIMESTAMP>/stage1_nn/last.pth
# eval stage 2 → expect ~1031 (vs 950 pretrained)
pixi run python rl_isaaclab/scripts/eval.py \
  --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 --algorithm ProprioAdapt \
  --load_path logs/debug/<TIMESTAMP>/stage2_nn/last.pth \
  --output_json experiments/repro_ours_stage2.json
```

### 3.2 Our xhand M3A result (requires retrain — no checkpoint on this machine)

The sphere grasp cache is present (`cache/xhand_sphere_grasp_linspace_1.0-1.0-1.npy`).
Reproduce EXP-M3A-8 (~250 M steps; several hours on the laptop):
```bash
pixi run python rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Xhand-v0 --headless --num_envs 4096
pixi run tb        # tensorboard --logdir logs/  → watch reward climb to ~+54
```
Visualize whatever checkpoint results:
```bash
pixi run python rl_isaaclab/scripts/play.py \
  --task Isaac-Inhand-Rotate-Xhand-v0 --num_envs 16 \
  --load_path logs/debug/<TIMESTAMP>/stage1_nn/best.pth
```

### 3.3 Quick render of any pose (no training, no Gemini key needed)
```bash
pixi run python rl_isaaclab/scripts/visual_check.py --robot xhand --gravity --steps 60
# writes a PNG to experiments/screenshots/ — open it to see the grasp.
```

---

## 4. What's next (pointer)

The honest read: M1/M2 are finished wins; M3A got xhand to a *weak hold* but the
**flat palm-up grasp is the ceiling**. The next thread tilts the palm so the
object is cradled in the corner between palm and fingers (gravity holds it *in*
instead of letting it roll out). Parameterized and ready to sweep — see
**`roadmap/M3B_angled_palm.md`**.
