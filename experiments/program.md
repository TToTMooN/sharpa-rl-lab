# Research Program

## Current Objective
Reproduce SHARPA RL published results for in-hand object rotation on the SharpaWave hand.

## Phase 1: Environment Bootstrap (current)
1. Get Isaac Lab installed and working on RTX 5090
2. Generate grasp cache successfully
3. Run a short training (1000 steps) to verify the pipeline works end-to-end
4. Visualize pretrained checkpoints to confirm environment correctness

## Phase 2: Baseline Reproduction
1. Train full PPO policy (300M steps) with default hyperparameters
2. Compare reward curves against expected behavior (rotation reward should climb steadily)
3. Run ProprioAdapt distillation on the trained policy
4. Evaluate both stage 1 and stage 2 policies

## Phase 3: Understanding & Ablation
1. Ablate key components: gravity curriculum, tactile sensing, privileged info
2. Test different object scales and masses
3. Profile GPU utilization and training throughput
4. Document what each component contributes to final performance

## Phase 4: Extension (future)
- Try different objects beyond cylinder
- Experiment with different reward shaping
- Test sim2real transfer improvements

## Constraints
- Do NOT modify `prepare.py`-equivalent files (asset definitions, sensor maps)
- Keep evaluation metrics consistent across experiments
- Always run headless for training experiments
- Log every experiment, successful or not

## Stopping Criteria
- Phase 1: Pipeline runs without errors
- Phase 2: Reward matches or exceeds pretrained checkpoint performance
- Phase 3: Clear understanding of each component's contribution
