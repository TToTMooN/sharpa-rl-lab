# M2: Ablation & Distillation Gap

**Goal**: Close the multi-scale distillation gap (EXP-011: our 878 vs pretrained 950) and understand which components of the SHARPA RL recipe are load-bearing.

**Depends on**: M1 (trained baselines exist for comparison)

**Key finding from M1**: Multi-scale distillation training reward plateaued at 186 (vs 992 for single-scale). Our distilled stage-2 eval is 7.6% below pretrained. Stage-1 PPO is fine. Distillation is the bottleneck.

## Phase A — Investigate the distillation gap (priority 1)

Small targeted experiments, cheap (each ~30-90 min):

| # | Experiment | Change | Question |
|---|------------|--------|----------|
| M2-1 | Longer distill | Run full 1B steps, snapshot every 100M, eval each | Does distill improve if we just wait? (EXP-010 already ran 1B and plateau'd — should verify via eval snapshots) |
| M2-2 | LR sweep | 1e-3, 3e-4 (baseline), 1e-4 | Is 3e-4 the right LR for multi-scale? |
| M2-3 | Warmup from best.pth not last.pth | Use `stage1_nn/best.pth` | Does starting from best vs last checkpoint help? |
| M2-4 | Expose scale in priv_info | Add object scale as 9th priv dim | Does adapt_tconv need explicit scale info? |
| M2-5 | Action-matching distill | Replace MSE(e, e_gt) with MSE(π(e), π(e_gt)) | Is embedding-matching suboptimal? |
| M2-6 | DAgger | Use PPO teacher actions instead of adapt_tconv rollouts | Is student-rollout dynamics the issue? |

**Deliverable**: One of these should close the gap to within ±2% of pretrained stage-2.

## Phase B — Component ablations on single-scale baseline

Use our best single-scale result (EXP-003, 1272 stage-1, 1137 stage-2) as reference. Train each ablation for 100M steps (~30 min) to keep costs down.

| # | Experiment | What to disable/change | Expected impact |
|---|------------|----------------------|-----------------|
| M2-7 | No gravity curriculum | `gravity_curriculum = False` | Slower convergence, fragile policy |
| M2-8 | No tactile sensing | Zero out contact sensor obs | Major reward drop — quantifies tactile value |
| M2-9 | No privileged info | `priv_info = False` | Slower PPO, direct critic comparison |
| M2-10 | No domain randomization | Disable all `randomize_*` flags | Good sim, bad sim2real |
| M2-11 | Smaller env sweep | 1024 / 4096 / 8192 / 16384 | FPS + sample efficiency curve |
| M2-12 | Short prop_hist | 1 / 3 / 10 / 30 frames | Find minimum temporal context |
| M2-13 | Stronger external force | `force_scale` 0, 0.5×, 1× (default), 2× | Robustness vs difficulty |

## Phase C — Reward structure decomposition

Understand what each of the 6 reward terms contributes:
1. `rotate_reward` — angular velocity on Z axis (primary signal)
2. `object_linvel_penalty` — penalize translation
3. `pos_diff_penalty` — hand joint deviation from default pose
4. `torque_penalty` — L2 torque
5. `work_penalty` — `(torque · vel)²`
6. `object_pos_diff` — inverse distance from hand center

Experiments:
- Zero each penalty in turn; compare stage-1 reward curves
- Ablate `rotate_reward` alone — does it converge without penalties?
- Record rotate_reward, yaw, torque separately (already in extras)

## Phase D — Profiling (cheap, parallel)

| Metric | Tool | Purpose |
|--------|------|---------|
| GPU util breakdown | `nvidia-smi dmon` during training | Find bottleneck |
| FPS vs num_envs | Sweep 1024–32768 | Throughput curve (for M3 planning) |
| VRAM scaling | Same sweep | RTX 5090 ceiling |
| CPU perf governor | `cpupower frequency-set -g performance` | Verify powersave is hurting us |

## Deliverables

- Close distillation gap (or document why not)
- Ablation table in experiments.tsv with reward-vs-ablation comparison
- Updated EXPERIMENTS.md narrative per group
- Knowledge for M3: which components are critical to preserve when extending

## Related Sharpa repos
- **sharpa-tacmap** — If M2-8 (disable tactile) shows tactile is load-bearing, tacmap integration becomes high priority for M4 sim2real
- **sharpa-tactile-sensor-assets** — ablating sensor coverage means modifying these maps
