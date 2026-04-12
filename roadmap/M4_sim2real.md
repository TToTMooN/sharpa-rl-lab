# M4: Sim2Real & Deployment

**Goal**: Deploy trained policies on physical SharpaWave hardware. Validate sim2real transfer quality.

**Depends on**: M1 (trained policy), M2 (understanding of domain rand), physical SharpaWave hand

## Prerequisites

- Physical SharpaWave hand + SharpaPilot calibration
- SharpaWave SDK installed (host computer or on-board tactile)
- Docker + nvidia-ctk setup (for host computer tactile, recommended)
- 3D printed cylinder (24mm radius, 60mm height recommended)

## Steps (planned)

1. Deploy pretrained checkpoint on real hardware first (verify setup)
2. Deploy our M1-trained policy and compare
3. Test domain randomization impact: compare policies with/without DR
4. Measure real-world metrics: rotation speed, drop rate, recovery behavior
5. Identify sim2real gap — what fails in real that works in sim?

## Relation to Sharpa ecosystem

- **SharpaPilot** is the control interface for deployment (calibration)
- **sharpa-tacmap** addresses the tactile sim2real gap specifically — may close gap we observe
- **sharpa-manus-sdk** — Manus teleop could collect demo trajectories for imitation learning on failures
- CraftNet's System 0 runs at ~100Hz — our deployment runs at 20Hz, room to explore faster control
- Real tactile feedback may differ from sim — Sharpa's 1000+ pixel tactile sensing is higher fidelity than the sim model in this repo
