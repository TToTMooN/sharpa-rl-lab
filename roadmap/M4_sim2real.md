# M4: Sim2Real & Full Hardware Reproduction (xhand)

**Goal**: Deploy trained policies on physical hardware and validate full sim2real pipeline.

**Primary target**: **RoboEra xhand** (we have real hardware + SDK locally)
**Secondary target**: SharpaWave (if physical hand becomes available)

**Depends on**:
- M1 ✅ (recipe validated on SharpaWave)
- M2 Phase A ✅ (distillation gap closed)
- **M3** (xhand baseline working in sim) — prerequisite
- **M2 Phase B/C/D** (reward decomposition + ablations) — strongly recommended before touching real hardware; we need to know which components are load-bearing before retuning them on real xhand
- Physical xhand + SDK

## Why xhand is the right M4 target

1. We have the hardware locally (user confirmed) + full SDK (`xhand_control_sdk_py`, `xhand_control_ros`, etc.)
2. It's 12 DOF (vs SharpaWave's 22) — smaller surface area for sim2real gap
3. xhand has 120-300 tactile sensors/fingertip — well-suited to our recipe's tactile feedback design
4. Finishing M4 on xhand proves the recipe is a **portable tool**, not a SharpaWave-specific demo

## Prerequisites checklist

- [ ] M3 complete: xhand working in sim, reward matches or beats our SharpaWave multi-scale baseline (1031 stage-2)
- [ ] M2 Phase B ablations on xhand (know which components are load-bearing)
- [ ] M2 Phase C reward decomposition (know which reward terms are load-bearing)
- [ ] xhand SDK installed: `xhand_control_sdk_py_x86_64_v117.zip` or ROS variant
- [ ] xhand driver installed: `xhand_v1.1.15.0_release_20250625.deb`
- [ ] Serial/USB connectivity test with real xhand
- [ ] 3D printed cylinder to match the scale range we trained on (0.4–0.6 scale)

## Steps (planned)

### Phase 1: Pipeline validation
1. Install xhand SDK + driver, verify real-hand communication
2. Deploy our best M3 stage-2 checkpoint on real xhand
3. Mirror the existing `deploy.py` script structure but swap SharpaWave SDK → xhand SDK
4. Sanity check: zero-action pose, joint position control, contact reading

### Phase 2: First real-hand rollouts
5. Run our xhand policy on 1 cylinder at the default scale
6. Measure: rotation speed, drop rate, full-episode survival
7. Compare qualitatively to sim behavior — what's different?

### Phase 3: Identify sim2real gaps
8. Categorize failures: dropped object, wrong contact pattern, wrong joint torques, timing issues
9. For each failure class, hypothesize root cause:
   - Tactile fidelity mismatch (sim simple model vs real 120-300 sensor array)
   - Dynamics mismatch (joint friction, backlash, actuator bandwidth)
   - Reward term mismatch (a term that mattered in sim but not in real)
10. Use M2 Phase B ablation insights to prioritize which gap to close first

### Phase 4: Iterate and close gaps
11. Retrain with tighter domain randomization on the identified gap
12. Optionally: integrate `sharpa-tacmap` for higher-fidelity tactile sim if tactile is the gap
13. Re-deploy, measure again, repeat

### Phase 5: Robustness checks
14. Vary cylinder size (test the multi-scale range)
15. Vary object material / friction
16. External disturbances (push the object)
17. Long-duration test (multi-minute rotation, not just 20s episodes)

## Key metrics for M4 success

| Metric | Target |
|--------|--------|
| First-try success rate (object not dropped in 20s) | ≥ 50% |
| Rotation speed (rad/s) | Comparable to sim eval |
| Multi-scale robustness | Works across 0.4–0.6 cylinder scales |
| Failure mode categorization | Documented, actionable |

## Relation to Sharpa ecosystem

- **sharpa-tacmap** addresses the tactile sim2real gap specifically — if tactile is the bottleneck, we port tacmap from their repo
- **sharpa-manus-sdk** — Manus teleop could collect demo trajectories for imitation learning on failure modes
- **CraftNet System 0** runs at ~100Hz — our current recipe is 20Hz control. Faster control may be needed for real xhand
- **MoDE-VLA** contact-rich tasks are far beyond this milestone, but our reward insights from M2 Phase C would be the foundation

## xhand-specific SDK references (all in ~/Downloads)

- `xhand_control_sdk_py_x86_64_v117.zip` — Python SDK (recommended for deploy.py port)
- `xhand_control_sdk_x86_64_v141.tar.gz` — C++ SDK
- `xhand_control_ros_x86_64_v123.tar.gz` — ROS1 bridge
- `xhand_control_ros2_x86_64_v132.tar.gz` — ROS2 bridge
- `XHAND Python SDK 使用文档1.2.10.pdf` — CN usage docs
- `XHAND Python SDK接口说明文档1.2.10.pdf` — CN interface specs
- `xhand_v1.1.15.0_release_20250625.deb` — driver

## Open questions to resolve before M4

- Does xhand's tactile array expose per-sensor force in the SDK, or aggregated? (affects how we map sim obs to real obs)
- What's the max control rate for xhand? (affects our 20Hz decimation choice)
- Does xhand have an emergency stop / compliance mode we can use as safety fallback?
- Are there published xhand benchmarks we can compare to (rotation tasks, manipulation demos)?
