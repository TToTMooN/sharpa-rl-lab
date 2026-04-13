# M5: xhand Sim2Real Deployment

**Goal**: Deploy our M3-trained policies on physical RoboEra xhand. Prove the full pipeline (asset → sim training → distillation → real robot) works on a non-Sharpa hand.

**Depends on**:
- M3 (trained xhand policies in sim, stage-1 and stage-2)
- M2 Phase B/C/D strongly recommended — understand which components are load-bearing before retuning on real xhand
- **Physical xhand** ✅ (available locally)
- xhand SDK + drivers (available in `~/Downloads/`)

**Current status**: 🟡 **Blocked on M3 completion.** Real hardware is on the shelf, SDK archives are ready. Waiting for sim-trained policies.

## Prerequisites

- [ ] M3 complete: xhand stage-2 policy in sim with reward comparable to SharpaWave baselines
- [ ] xhand driver installed: `xhand_v1.1.15.0_release_20250625.deb`
- [ ] xhand Python SDK extracted: `xhand_control_sdk_py_x86_64_v117.zip`
- [ ] USB/serial connection to xhand verified
- [ ] Safety: E-stop or compliance mode available during deployment

## xhand-specific SDK references (all in ~/Downloads/)

- `xhand_control_sdk_py_x86_64_v117.zip` — Python SDK (for `deploy_xhand.py` port)
- `xhand_control_sdk_x86_64_v141.tar.gz` — C++ SDK
- `xhand_control_ros_x86_64_v123.tar.gz` — ROS1 bridge
- `xhand_control_ros2_x86_64_v132.tar.gz` — ROS2 bridge
- `xhand_v1.1.15.0_release_20250625.deb` — driver
- `XHAND Python SDK 使用文档1.2.10.pdf` — CN usage docs
- `XHAND Python SDK接口说明文档1.2.10.pdf` — CN interface specs

## Steps (to run after M3)

### Phase 1: Driver + SDK installation
1. Install xhand .deb driver package
2. Extract Python SDK, verify import works
3. Serial/USB connection test — read joint positions from real hand
4. Safely power-cycle test, verify E-stop

### Phase 2: Port deploy.py to xhand
5. Clone SharpaWave's `deploy.py` → `deploy_xhand.py`
6. Replace sharpa SDK imports with xhand SDK imports
7. Map xhand joint order to our trained policy's joint order (action_space=12)
8. Map xhand tactile readings to our 5-sensor observation format
9. Dry-run with zero actions — verify commands reach the hand safely

### Phase 3: Deploy M3 trained policies
10. Deploy our xhand stage-2 policy on real xhand
11. Cylinder rotation task with cylinder scales matching our trained range
12. Measure: rotation speed, drop rate, full-episode rate
13. Record video

### Phase 4: Iterate on sim2real gap
14. Categorize failures (as in M4)
15. Leverage M2 Phase B/C/D insights to decide which component to retune
16. Retrain on xhand with tighter DR or different reward weights
17. Re-deploy

### Phase 5: Cross-hand comparison
18. Run same task on SharpaWave (M4) — compare failure modes
19. Does the same sim2real technique work for both hands, or are the gaps different?
20. This is the real test of recipe portability

## Success metrics

| Metric | Target |
|--------|--------|
| First-try object retention on real xhand | ≥ 50% |
| Rotation speed | Comparable to our sim eval |
| Match between SharpaWave failure modes (M4) and xhand failure modes | Overlapping — suggests recipe-level sim2real gap |

## Open questions

- Does xhand SDK expose per-tactile-pixel data or aggregated force? (affects how we map sim obs to real obs; may need a retargeting step)
- xhand control rate — can we run 20Hz or do we need faster?
- Joint ordering convention in xhand URDF vs SDK — need to verify matches our trained policy
- xhand compliance / backdrivability in real vs sim — the URDF implies stiff joints but the hand is mechanically back-drivable

## Difference from M4 (SharpaWave sim2real)

| Aspect | M4 (SharpaWave) | M5 (xhand) |
|--------|----------------|-----------|
| Hand DOF | 22 | 12 |
| Pretrained baseline exists | Yes (Sharpa ships) | No — we train from scratch in M3 |
| Tactile model in sim | Simpler (10 sensors) | Simpler still (5 sensors) |
| Real tactile bandwidth | 1000+ pixels/fingertip | 120-300 sensors/fingertip |
| Success = | Match Sharpa's published results | Match our own M3 sim eval |

M5 is both a proof-of-portability and a standalone deployment — it does not depend on M4 succeeding.
