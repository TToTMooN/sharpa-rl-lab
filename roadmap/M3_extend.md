# M3: Extend Tasks & Objects

**Goal**: Go beyond the default cylinder rotation demo. Test generalization of the pipeline to new objects, tasks, and configurations.

**Depends on**: M1 (working baseline), M2 (understanding of components)

## Possible extensions

### New objects
- Different shapes: sphere, cube, irregular objects
- Different sizes: smaller/larger than default 24mm radius cylinder
- Different materials: varying friction coefficients
- Deformable objects (if Isaac Lab supports)

### New tasks
- **Multi-axis rotation**: Rotate around X/Y in addition to Z
- **Reorientation**: Reach a target orientation, not continuous rotation
- **Object handoff**: Transfer between fingers
- **Precision placement**: Rotate to target angle and hold

### Architecture experiments
- Replace MLP with transformer for temporal processing
- Try different action spaces (position vs torque control)
- Asymmetric actor-critic with different obs spaces
- Reward shaping experiments

## Relation to Sharpa's broader work

- **CraftNet** uses these low-level policies as building blocks — extending the task library directly enables more CraftNet capabilities
- **MoDE-VLA** tackles contact-rich tasks (apple peeling, gear assembly) — understanding policy generalization here informs that work
- **Tacmap** (NVIDIA partnership) needs diverse tasks for validation
