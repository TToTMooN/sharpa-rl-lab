# M2: Ablation & Distillation Gap

**Status**: **Phase A complete** (distillation gap closed). **Phase B/C/D deferred to post-M3** (see bottom of file).

**Goal**: Close the multi-scale distillation gap (EXP-011: our 878 vs pretrained 950) and understand which components of the SHARPA RL recipe are load-bearing.

**Depends on**: M1 (trained baselines exist for comparison)

**Key finding from M1**: Multi-scale distillation training reward plateaued at 186 (vs 992 for single-scale). Our distilled stage-2 eval is 7.6% below pretrained. Stage-1 PPO is fine. Distillation is the bottleneck.

## Phase A — Investigate the distillation gap ✅ CLOSED

**Result**: EXP-014 scale-aware stage-2 gets **1031.46** mean reward vs pretrained **950.38** (+8.5%). Distillation gap closed in one targeted experiment (M2-4).

| # | Experiment | Change | Question | Status |
|---|------------|--------|----------|--------|
| M2-1 | Longer distill | Run full 1B steps | Does distill improve with more time? | ❌ Not needed (EXP-010 showed plateau at 186) |
| M2-2 | LR sweep | 1e-3, 3e-4, 1e-4 | Right LR for multi-scale? | ⏸ Skipped — M2-4 solved it |
| M2-3 | Warmup from best.pth | vs last.pth | Does init matter? | ⏸ Skipped |
| **M2-4** | **Expose scale in priv_info** | priv_info_dim 8→9 | **Does adapt_tconv need explicit scale?** | ✅ **CLOSED THE GAP** (EXP-012/013/014) |
| M2-5 | Action-matching distill | MSE(π(e), π(e_gt)) | Is embedding-matching suboptimal? | ⏸ Skipped |
| M2-6 | DAgger | Teacher actions | Is student-rollout the issue? | ⏸ Skipped |

**Root cause** (EXP-014 narrative in EXPERIMENTS.md): `env_mlp(priv_info)` target was scale-invariant but `adapt_tconv(proprio_hist)` input was scale-dependent → MSE loss had ambiguous target across 8 scales → adapt_tconv couldn't converge. One-line fix: `self.priv_info_buf[:, 8] = self.env_scales`.

**Phase A deliverable**: ✅ M1 multi-scale gap closed. Both configs (single-scale and multi-scale) now reproduce + exceed pretrained.

---

## Phase B/C/D — DEFERRED until between M3 and M4

**Rationale for deferring**: Phase A solved the immediate problem (distillation gap). The remaining phases are about *understanding* the recipe, which is most valuable right before M4 (sim2real on xhand). Running ablations now would delay M3 without a clear payoff; running them between M3 and M4 means:
1. We'll know the recipe works on a second hand (xhand) before ablating
2. Ablations can be run on BOTH SharpaWave and xhand baselines → stronger conclusions
3. Reward design understanding is load-bearing for M4 (real hardware tuning)
4. M4 reproduction on xhand needs these insights — guessing reward weights on new hardware is risky

**Re-entry point**: After M3 delivers a working xhand baseline. Label experiments M2B-*, M2C-*, M2D-* to distinguish from M2 Phase A (which is done).

### Phase B — Component ablations (DEFERRED)

Use our best single-scale result (EXP-003, 1272 stage-1, 1137 stage-2) as reference. Train each ablation for 100M steps (~30 min) to keep costs down. Repeat on xhand once available.

| # | Experiment | What to disable/change | Expected impact | M4 relevance |
|---|------------|----------------------|-----------------|-------------|
| M2B-1 | No gravity curriculum | `gravity_curriculum = False` | Slower convergence, fragile policy | High — curriculum needs retuning per hand |
| M2B-2 | No tactile sensing | Zero out contact sensor obs | Major reward drop — quantifies tactile value | Critical — tactile fidelity is the sim2real gap |
| M2B-3 | No privileged info | `priv_info = False` | Slower PPO, direct critic comparison | Medium |
| M2B-4 | No domain randomization | Disable all `randomize_*` flags | Good sim, bad sim2real | Critical — sim2real depends on DR |
| M2B-5 | Smaller env sweep | 1024 / 4096 / 8192 / 16384 | FPS + sample efficiency curve | Medium — planning hardware budgets |
| M2B-6 | Short prop_hist | 1 / 3 / 10 / 30 frames | Find minimum temporal context | High — xhand has different proprio bandwidth |
| M2B-7 | Stronger external force | `force_scale` 0, 0.5×, 1× (default), 2× | Robustness vs difficulty | High — real hand encounters disturbances |

### Phase C — Reward structure decomposition (DEFERRED — needed before M4)

Understand what each of the 6 reward terms contributes. **This is the most load-bearing for M4**: on xhand, reward weights will need retuning, and we should know which terms actually matter.

Reward terms in `sharpa_wave_env.py::compute_rewards`:
1. `rotate_reward` — angular velocity on Z axis (primary signal)
2. `object_linvel_penalty` — penalize translation
3. `pos_diff_penalty` — hand joint deviation from default pose
4. `torque_penalty` — L2 torque
5. `work_penalty` — `(torque · vel)²`
6. `object_pos_diff` — inverse distance from hand center

Experiments:
- M2C-1: Zero each penalty in turn, train 100M, compare reward curves and qualitative behavior
- M2C-2: `rotate_reward` only — minimum viable reward
- M2C-3: Sweep `rotate_reward_scale`, `torque_penalty_scale` to find sensitivity
- M2C-4: Decompose our EXP-012 tensorboard per-term curves retroactively (no new training, just analysis)

### Phase D — Profiling (cheap) — could do anytime

Low-cost, can slot in between M3 experiments while waiting.

| Metric | Tool | Purpose |
|--------|------|---------|
| GPU util breakdown | `nvidia-smi dmon` during training | Find bottleneck |
| FPS vs num_envs | Sweep 1024–32768 | Throughput curve (for M3 planning) |
| VRAM scaling | Same sweep | RTX 5090 ceiling |
| CPU perf governor | `cpupower frequency-set -g performance` | Verify powersave is hurting us |

### Phase B/C/D deliverables (when revisited)

- Table of ablation results on both SharpaWave + xhand in `experiments.tsv`, labeled as M2B-*/M2C-*/M2D-*
- Narrative per group in EXPERIMENTS.md
- **Reward design guide** for M4: which terms can be kept as-is, which need retuning per hand
- Component criticality ranking → inform what's safe to strip for real hardware deployment

## Related Sharpa repos
- **sharpa-tacmap** — If M2B-2 (disable tactile) shows tactile is load-bearing, tacmap integration becomes high priority for M4 sim2real
- **sharpa-tactile-sensor-assets** — ablating sensor coverage means modifying these maps
