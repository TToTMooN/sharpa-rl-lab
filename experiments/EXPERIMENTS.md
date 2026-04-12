# Experiment Log

Structured metrics live in [`experiments.tsv`](experiments.tsv) (machine-readable).
This file has the narrative: hypotheses, detailed findings, and decisions.

**Primary metric**: mean episode reward
**Hardware**: RTX 5090 (Blackwell, sm_120), AMD Ryzen 9 9900X, 62GB RAM
**Framework**: Isaac Lab 2.3.2.post1 + Isaac Sim 5.1.0.0 + PyTorch 2.7.0+cu128

---

## EXP-000: Environment Bootstrap
**Hypothesis**: Pixi + pip can manage Isaac Lab on RTX 5090 without conda.
**Outcome**: Works. Key finding: Isaac Lab pip pulls PyTorch cu126 by default — must override with cu128 for sm_120.
**Install sequence**: pixi (Python 3.11) → pip isaaclab[isaacsim,all] → pip torch+cu128 → pip -e .

---

## EXP-001: Grasp Cache Generation
**Hypothesis**: Isaac Lab physics sim works on RTX 5090 Blackwell.
**Outcome**: 50k grasps in ~19 min. Vulkan backend works. CPU powersave mode and IOMMU enabled — could hurt perf.

---

## EXP-002: Smoke Test PPO Training
**Hypothesis**: Full PPO pipeline runs after fixing compat bugs.
**Bugs found & fixed**:
1. `import gym` in experience.py — unused, removed (gymnasium is correct)
2. Missing `termcolor` dep — installed
3. `torch.zeros(N,1,3)` without `device=self.device` in sharpa_wave_env.py:196 — caused warp kernel crash

**Outcome**: 8.1k FPS at 1024 envs. Reward started at 1.48, peaked 2.69, dipped with gravity curriculum. Pipeline validated.

---

## EXP-003: Full-Scale PPO Training
**Hypothesis**: Default hyperparams with 16384 envs will produce a policy comparable to pretrained/0.5-0.5-1.pth.
**Outcome**: 300M steps completed in **91 minutes** at 55k FPS. Best reward **1130.05**, final mean **1088.09**.
Reward trajectory: 0.58 → 115 (9M) → 480 (14M) → 795 (65M) → 928 (131M) → 1027 (262M) → 1130 (best).
Gravity curriculum caused expected dips mid-training. Checkpoints saved every 500 iterations.
**Checkpoints**: `logs/debug/2026-04-11_21-50-19/stage1_nn/{best,last}.pth`
**Next**: EXP-004 — evaluate our trained policy vs pretrained checkpoint.

---

## EXP-004: Evaluate Our Stage-1 PPO Policy
**Hypothesis**: Our EXP-003 checkpoint (PPO stage 1) performs comparably to pretrained policy.
**Changes**: Wrote `rl_isaaclab/scripts/eval.py` — quantitative eval script with episode reward tracking, success rate, JSON output.
**Outcome**: Mean reward **1272.79** ± 127.0 over 256 episodes, **99.6% full-episode rate** (object retained almost always). Median 1288.75. Policy is strong.
**Checkpoint**: `logs/debug/2026-04-11_21-50-19/stage1_nn/best.pth`
**JSON**: `experiments/exp004_eval_ours.json`

---

## EXP-005: Evaluate Pretrained Checkpoint
**Hypothesis**: Pretrained `0.5-0.5-1.pth` is a reasonable upper-bound reference for our reproduction.
**Discovery**: The pretrained checkpoint is a **ProprioAdapt stage-2** model — has `adapt_tconv` layers that PPO's ActorCritic doesn't instantiate. Updated eval.py to handle both PPO and ProprioAdapt inference (different input dicts: `priv_info` vs `proprio_hist`).
**Outcome**: Mean reward **1040.63** ± 99.1 over 256 episodes, 100% full-episode rate. Median 1052.64. Object retained perfectly but reward lower than our stage-1 PPO.
**Interpretation**: Stage-2 policies use only proprioceptive history (no privileged info) — designed for deployment, so naturally lower ceiling. **Not a fair comparison** against our stage-1 PPO.
**JSON**: `experiments/exp005_eval_pretrained.json`

### Comparison table
| Policy | Mean | Std | Min | Max | Median | Full-ep |
|--------|------|-----|-----|-----|--------|---------|
| EXP-004 ours PPO stage1 | **1272.79** | 127.0 | 26.2 | 1327.6 | 1288.75 | 99.6% |
| EXP-005 pretrained stage2 | 1040.63 | 99.1 | 113.4 | 1168.5 | 1052.64 | 100% |

**Next**: EXP-006 — distill our stage-1 into a stage-2 ProprioAdapt policy for fair comparison with pretrained.
