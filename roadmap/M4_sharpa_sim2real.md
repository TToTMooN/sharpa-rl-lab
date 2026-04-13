# M4: SharpaWave Sim2Real Deployment

**Goal**: Deploy our M1+M2A trained policies on physical SharpaWave hardware. Validate sim2real transfer quality for the original target hand.

**Depends on**:
- M1 ✅ (trained SharpaWave policies in sim — both single-scale and multi-scale, stage-1 and stage-2)
- M2A ✅ (distillation gap closed, scale-aware priv_info)
- M2 Phase B/C/D recommended but not strictly required — informs which components matter for sim2real tuning
- **Physical SharpaWave hand** — on order, arrival pending

**Current status**: 🟡 **Blocked on hardware arrival.** SharpaWave hand was ordered and is in transit. Everything else (software, SDK scripts, docker setup) can be pre-staged.

## Prerequisites

- [ ] Physical SharpaWave hand delivered
- [ ] SharpaPilot calibration tool installed
- [ ] SharpaWave SDK installed (host-computer tactile mode preferred — higher frame rate)
- [ ] Docker + nvidia-ctk setup (for host computer tactile)
- [ ] 3D printed cylinder (24mm radius, 60mm height)
- [ ] Workspace with bench for hand mounting + object retrieval

## Checkpoints ready for deployment

From M1/M2A we have:
- **`logs/debug/2026-04-11_21-50-19/stage2_nn/best.pth`** — single-scale distilled (eval 1137.66, EXP-007)
- **`logs/debug/2026-04-12_16-12-33/stage2_nn/best.pth`** — multi-scale distilled with scale-aware priv_info (eval 1031.46, EXP-014)

Both are stage-2 ProprioAdapt policies, directly deployable via `deploy.py`.

## Steps (to run once hardware arrives)

### Phase 1: Sanity + baseline
1. Install SharpaWave SDK, verify hand communication (zero-action pose, joint position targets)
2. Deploy the **shipped pretrained** checkpoint (`pretrained/0.5-0.5-1.pth`) first — baseline for what Sharpa reports
3. Measure: rotation speed, drop rate, full-episode survival rate
4. Record video for comparison

### Phase 2: Deploy our policies
5. Deploy single-scale stage-2 (our EXP-007)
6. Compare to shipped pretrained (both trained for same config) — should match or beat
7. Deploy multi-scale stage-2 (our EXP-014) — try across different cylinder sizes

### Phase 3: Sim2real gap analysis
8. Categorize failures: dropped object, tactile mismatch, actuator saturation, timing/latency
9. Quantify tactile sim2real gap: read real tactile data during a rollout, compare to sim tactile pattern on the same trajectory
10. If tactile is the gap → integrate `sharpa-tacmap` (Sharpa+NVIDIA high-fidelity tactile sim) + retrain

### Phase 4: Iterate
11. Use M2 Phase B ablation results to decide which domain randomization to tighten
12. Retrain, re-deploy, measure again

### Phase 5: Robustness
13. Vary cylinder size (test multi-scale robustness across 0.4–0.6 range)
14. Vary object material / friction
15. External disturbances (push the object during rotation)
16. Long-duration test (multi-minute rotation)

## Success metrics

| Metric | Target |
|--------|--------|
| First-try object retention (20s episode) | ≥ 50% |
| Rotation speed | Comparable to sim eval |
| Multi-scale robustness | Works across 0.4–0.6 cylinder scales |
| Match to Sharpa's published real-world performance | ≥ 80% of their reported numbers |

## Relation to Sharpa ecosystem

- **SharpaPilot**: Control interface for deployment + calibration
- **sharpa-tacmap**: NVIDIA-partnered high-fidelity tactile sim — may be needed to close the tactile gap
- **sharpa-manus-sdk**: Manus teleop for collecting failure-mode demonstrations (optional)
- **CraftNet System 0**: Runs ~100Hz on real hand; our sim trains at 20Hz decimation. May need faster control during deployment.
- **Shipped real-world benchmarks**: The repo's `resources/real.gif` shows Sharpa's in-hand rotation on real hardware — target visual quality.

## Open questions to resolve

- What tactile sensor bandwidth does SharpaWave SDK actually expose (per-pixel vs aggregated)?
- What's the effective control rate — can we run faster than 20Hz on real hand?
- Does the SDK support async sensor reads while applying action commands, or is it blocking?
- Does the real hand have an emergency stop / compliance mode for safety fallback during learning?

## Pre-hardware work we can do now

- Port `deploy.py` from SharpaWave SDK v0 references to current SDK version once SDK is installed
- Verify docker configuration works with current setup
- Pre-stage all offline prep in a dedicated branch so merging is clean when hardware lands
