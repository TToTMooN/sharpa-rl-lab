# Setup Environment

Bootstrap the development environment using pixi.

## Instructions

1. Run `pixi install` to create the environment
2. Verify Isaac Lab is importable: `pixi run python -c "import isaaclab; print(isaaclab.__version__)"`
3. Verify PyTorch + CUDA: `pixi run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
4. Verify the project is installed: `pixi run python -c "import rl_isaaclab"`
5. If any step fails:
   - Check error messages carefully
   - For RTX 5090 issues: try PyTorch nightly, check CUDA version
   - For Isaac Lab issues: check Python version (needs 3.11), check pip index URLs
   - Log the issue and workaround in experiments/EXPERIMENTS.md as EXP-000
6. Report the environment status

$ARGUMENTS - Optional: specific issue to troubleshoot
