# Log Experiment Result

Log the results of a completed experiment to both the TSV index and the narrative log.

## Instructions

Given experiment details (either from $ARGUMENTS or from the most recent run):

1. Read `experiments/experiments.tsv` to determine the next experiment ID
2. Append a TSV row with these columns (tab-separated):
   `id  date  milestone  description  num_envs  max_steps  duration_min  fps  best_reward  mean_reward_final  vram_gb  status  decision  notes`
   Use `-` for unavailable fields. Keep description short and snake_case.
   Set milestone to the relevant roadmap milestone (e.g. M1, M2). Check `roadmap/README.md` for active milestones.
3. Append a narrative section to `experiments/EXPERIMENTS.md`:
   ```
   ## EXP-{ID}: {title}
   **Hypothesis**: {what we expected and why}
   **Outcome**: {what actually happened, key findings}
   ```
4. If KEEP: ensure changes are committed with message "EXP-{ID}: {description}"
5. If REVERT: revert code changes but keep both log entries

$ARGUMENTS - Experiment details: hypothesis, results, and decision
