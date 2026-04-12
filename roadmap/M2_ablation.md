# M2: Understand & Ablate Components

**Goal**: Understand what each component of the SHARPA RL pipeline contributes to final performance. Build intuition for what to change when extending.

**Depends on**: M1 (need a trained baseline to ablate against)

## Ablation experiments (planned)

| Experiment | What to disable/change | Expected impact | Why it matters |
|-----------|----------------------|-----------------|----------------|
| No gravity curriculum | Set `gravity_curriculum = False` | Slower convergence, fragile policy | Curriculum is key training trick |
| No tactile sensing | Disable contact sensors in obs | Major performance drop | Core of Sharpa's approach |
| No privileged info | Set `priv_info = False` | Worse critic, slower training | Tests if priv info is load-bearing |
| No domain randomization | Remove event randomizers | Good sim perf, bad real transfer | Quantify sim2real gap contribution |
| Fewer envs | 4096, 8192 vs 16384 | Slower wall-clock, possibly worse sample efficiency | Understand scaling behavior |
| Different object mass | Sweep 0.01–0.25 kg range | Harder at extremes | Map difficulty landscape |
| Tactile history length | 1, 3, 5, 10 frames | Diminishing returns after ~3 | Find minimal tactile context |
| External force scale | 0x, 0.5x, 1x, 2x | More forces = more robust but harder | Robustness vs. difficulty tradeoff |

## Profiling experiments (planned)

| Metric | Tool | Purpose |
|--------|------|---------|
| GPU utilization breakdown | `nvidia-smi dmon` | Where is time spent? |
| Physics vs. RL time | Training log timestamps | Bottleneck identification |
| Memory scaling | Sweep num_envs | Find VRAM ceiling on RTX 5090 |
| FPS vs num_envs | Sweep 1024–32768 | Throughput curve |

## Deliverables

- Table of ablation results in experiments.tsv
- Narrative in EXPERIMENTS.md explaining what each component does
- Updated program.md with refined understanding

## Related Sharpa repos
- **sharpa-tacmap** — Their high-fidelity tactile sim framework. If we enable ablation of tactile fidelity (low-res vs high-res), we may want to pull in tacmap integration here.
- **sharpa-tactile-sensor-assets** — Static sensor data already in `assets/tactile_ha4_map/`. Ablating sensor coverage/noise means modifying these.
