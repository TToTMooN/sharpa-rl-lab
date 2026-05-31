# M3B: Angled (non-flat) palm grasp for xhand

**Branch**: `m3-curriculum`
**Started**: 2026-05-31
**Parent thread**: `roadmap/M3A_curriculum.md` (got xhand to a weak +54 hold)

## Replan — what's right, what's wrong (assessment from scratch)

### What's RIGHT in the existing plan (keep)
- **M1/M2 are done and correct.** Beating the SHARPA baseline + the one-line
  `priv_info` scale fix are real, verified wins. Don't re-litigate them.
- **The hand-agnostic refactor is the right abstraction.** Per-hand config +
  shared algorithm is exactly how you port a recipe to a new embodiment.
- **The gravity curriculum + alive bonus are good tools** and should be reused.
- **The diagnosis in M3A is correct**: the −1000 plateau was `work_penalty`
  dominating + drop-on-reset. Zeroing the work penalty was the right call.

### What's WRONG / mis-prioritized (fix)
1. **We spent ~30 pose iterations and 8 curriculum experiments forcing a FLAT
   palm-up grasp to work.** The flat palm is the wrong primitive for xhand. With
   the palm facing straight up, gravity is *perpendicular* to the cradle and the
   only thing stopping the object escaping through the thumb-opposite gap is
   friction. We then leaned on huge friction (5.0) and an alive bonus to paper
   over it. The +54 result is a *marginal hold dominated by the alive bonus*,
   not rotation. **This is treating a geometry problem with reward shaping.**
2. **The sphere switch was a symptom-fix, not a cure.** We moved cylinder→sphere
   because the flat cup couldn't cage a cylinder. A sphere in a flat cup is still
   one nudge from rolling out.
3. **`reset_height` / alive-bonus thresholds are hand-tuned magic numbers** tied
   to the specific flat pose (z≈0.6). They'll all move when the pose changes, so
   they shouldn't be over-fit now.

### The user's insight (the actual fix)
> "the hand palm doesn't have to be flat when we hold things, it can have an angle."

Correct, and it's the crux. A human cupping a ball **tilts** the hand ~30–60°
so the object rests in the *corner* between the palm and the curled fingers.
Gravity then presses the object **into** that corner (palm + finger wall both
push back) instead of pulling it *off* a flat tray. This converts the grasp from
"balance on a ring of fingertips" (unstable, needs friction) to "wedge in a
corner" (stable, force-closure-ish even at low friction).

This should:
- make the cached grasps *actually stable* (survive gravity without 5.0 friction
  and without the alive bonus crutch),
- let `rotate_reward` — not the alive bonus — drive the policy,
- and be closer to how the **real xhand** will hold the object this week.

## Implementation (done in this commit)
- `palm_pose.py`: `palm_quat(roll, pitch, yaw)` — palm orientation as sweepable
  XYZ-Euler degrees, matching `isaaclab.utils.math.quat_from_euler_xyz`.
  `palm_quat(0, -90, 0)` reproduces the legacy flat pose **exactly** (verified).
- `xhand_env_cfg.py` and `xhand_grasp_env_cfg.py` now derive `hand_init_pose`
  from a `palm_euler_deg` field (default `(0, -90, 0)` = no behavior change).
  **The two MUST stay in sync** — the cache is generated at the grasp cfg's
  orientation and consumed at the train cfg's orientation.

## Procedure (once Isaac Lab finishes installing)

### Step 1 — Visually find the tilt (fast, no training, no Gemini key)
Render candidate tilts and **look at the PNG** (`experiments/screenshots/`):
```bash
pixi run python rl_isaaclab/scripts/visual_check.py --robot xhand --gravity --steps 80 \
  --hand_rot <w> <x> <y> <z>            # quaternion from palm_quat(roll,pitch,yaw)
```
Sweep, e.g. (roll = tilt the cup sideways so the open gap faces up; pitch toward
0 = palm faces more forward):
| roll | pitch | yaw | quat (w,x,y,z) | intent |
|------|-------|-----|----------------|--------|
| 0    | −90   | 0   | 0.7071, 0, −0.7071, 0      | legacy flat (baseline) |
| 25   | −90   | 0   | 0.6903, 0.153, −0.6903, 0.153 | tilt cup toward pinky |
| 0    | −65   | 0   | 0.8434, 0, −0.5373, 0      | palm up-and-forward |
| 20   | −70   | 0   | 0.8067, 0.142, −0.5649, 0.0996 | combined corner cradle |
Pick the orientation where the object visibly settles into the palm/finger
corner and stays put under gravity. Set BOTH `palm_euler_deg` fields to it, and
move the object init pos (`object_cfg.init_state.pos`) to where it rests.

### Step 2 — Regenerate the grasp cache at the new angle
```bash
pixi run python rl_isaaclab/scripts/gen_grasp.py \
  --task Isaac-Inhand-Rotate-Grasp-Xhand-v0 --headless --num_envs 4096
```
Success signal to watch in the log: `cond2` (≥2 fingertip forces) should pass on
a much larger fraction of envs than the flat pose's 9–20/1024, and the cache
should fill **without** needing friction 5.0. If it fills fast and stable, the
angle is right.

### Step 3 — Smoke train, then full train
```bash
pixi run python rl_isaaclab/scripts/train.py \
  --task Isaac-Inhand-Rotate-Xhand-v0 --headless --num_envs 4096 --max_agent_steps 5000000
```
**Decisive test**: with a genuinely stable corner cradle, try turning the
**alive bonus DOWN** (`alive_bonus_scale` 5→1) and **rotate_reward UP**. If reward
stays positive and `rotate_reward` (yaw) climbs, the grasp is real and we have
rotation — the thing M3A could not get. Then ramp gravity as before and distill.

## Success criteria
- **M3B succeeds if**: grasp cache fills at the angled pose with low friction
  (≤1.0) AND a training run shows positive `rotate_reward` (not just alive bonus)
  under full gravity. That is the first *real* in-hand rotation on xhand.
- **M3B informs the real hand**: the chosen tilt is the pose to start the
  physical xhand grasp from this week (M5).

## Risks
- The tilt that's stable in sim may differ from the real hand's best tilt;
  iterate on the real hand too.
- If even the corner cradle won't rotate (object just sits), the limiting factor
  is xhand's low finger count for rotation, not the hold — fall back to a smaller
  object or accept "stable hold + small reorient" as the xhand deliverable.

---

## 2026-05-31 findings (laptop session) — the diagnosis got sharper

Picked this up on the fresh laptop. Installed the stack, **headless training works**
(reproduced M3A's +120 zero-g hold in a smoke run). Two hard constraints + one
hard finding emerged:

### Constraint 1 — rendering is broken on this laptop (no visual iteration)
`--enable_cameras` segfaults in the RTX renderer / `create_new_stage`
(known RTX 5090 issue, aggravated by IOMMU + multi-GPU P2P). So the VLM
`visual_check` image loop is **unavailable here** — we are blind in sim. Tried
`CUDA_VISIBLE_DEVICES=0`; still crashes. Headless physics (no camera) is fine.

### Constraint 2 — so I iterate on physics signals, not images
Built two camera-free tools: `scripts/palm_probe.py` (drop sphere onto a pose,
sweep palm tilt × tip-curl) and `scripts/cache_hold_check.py` (replay the real
grasp cache under full gravity).

### Hard finding — the xhand fingertips never actually grip the object
- **Faithful signal (real env):** in the M3A smoke at g=0, the reward debug shows
  `n_touch=0.000` while `alive_frac=0.906`. The sphere is "alive" only because
  gravity is zero — **zero fingertips register contact force on it.** The cache
  encodes *proximity*, not *grip*.
- **Grasp criteria are too lax:** the cache accepts ≥2 fingertips at >0.2 N. A
  0.05 kg sphere weighs 0.49 N, so a 0.2 N feather-touch on two tips cannot hold
  it — the cache fills with non-holding momentary contacts.
- **M3A-8 in context:** the +128→+54 decay as gravity ramped, and `alive_frac`
  0.92→0.81, means the *trained* policy actively balances the sphere ~81 % of the
  time at full gravity but with ~0 rotation reward. It's a wobbly actively-held
  balance, **not a force-closure grasp**, and it leaves no stability budget for
  rotation. That is the real ceiling — and it's a grasp/geometry problem.

(Caveat: my standalone probes are less trustworthy than the env signal — without
a ground plane / fixed base they overstate the fall. The load-bearing evidence is
the env's own `n_touch=0` and the M3A-8 curve, both faithful.)

### Consequence — the palm angle alone won't fix it, and I can't see to tune it
Tilting the palm helps gravity press the object into a corner, but the corner
only exists if the *fingertips close onto the object*. They don't, and I can't
visually find a pose that makes them, because rendering is dead on this box. The
two real unblocks:

- **(A) Real-hand prior (recommended; hardware is here this week).** Back-drive
  the physical xhand to cradle the object, read the 12 joint angles via the xhand
  Python SDK, estimate the palm angle, and seed the cache from that *known-good*
  grip via `scripts/synth_xhand_cache.py` (already takes a pose → perturbed cache)
  + set `palm_euler_deg`. A human-found grip sidesteps the whole blind-search
  problem. Needs: install `xhand_v1.1.15.0…deb` + `xhand_control_sdk_py…zip`
  (were in `~/Downloads/`, absent on this laptop), connect USB, read state.
- **(B) Fix rendering, then run the VLM/visual pose loop.** Needs driver/IOMMU
  work on this laptop (or use the old desktop where the committed screenshots
  prove rendering worked). Then `palm_euler_deg` sweeps become eyeball-fast.

A third, cheaper hedge if neither is ready: **change the task to match xhand's
morphology** — rest the object on an *upward open palm* (object on the palm
surface, fingers as low side-walls) and rotate about vertical, rather than
demanding a force-closure cage. Less ambitious but may actually rotate.
