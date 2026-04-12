# Run Experiment (Autoresearch Loop)

Execute one iteration of the autoresearch experiment loop. This is the core workflow.

## Instructions

You are running an autoresearch-style experiment. Follow this exact sequence:

### 1. Read State
- Read `experiments/program.md` for current research direction
- Read `experiments/experiments.tsv` for structured metrics from prior runs
- Read `experiments/EXPERIMENTS.md` for narrative context and findings
- Check git log for recent changes
- Identify what to try next based on prior results

### 2. Hypothesize
- State a clear hypothesis: "I expect X because Y"
- Identify the single change to make
- Predict the expected outcome

### 3. Modify
- Make the targeted change (config, code, or hyperparameter)
- Keep changes minimal and isolated — one variable at a time
- Stage the change with git

### 4. Run
- Execute the training/evaluation command
- Use `--headless` always
- For quick iteration, use short runs (smoke-test or reduced max_agent_steps)
- Record the exact command used

### 5. Evaluate
- Check the output metrics (reward, loss, etc.)
- Compare against previous experiments
- Determine: improved, degraded, or neutral?

### 6. Log
- Append a row to `experiments/experiments.tsv` with structured metrics
- Append a narrative section to `experiments/EXPERIMENTS.md` with hypothesis + outcome
- If improved: commit the change with message "EXP-{ID}: {description}"
- If not improved: revert code changes, but keep both log entries

### 7. Next
- Based on results, update hypothesis for next experiment
- Continue the loop or stop if stopping criteria met

$ARGUMENTS - Optional: description of what to experiment with
