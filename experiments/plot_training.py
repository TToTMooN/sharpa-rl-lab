#!/usr/bin/env python3
"""Parse a training log and print training curve data as TSV.

Usage: python plot_training.py experiments/exp003_full_train.log [--bin 1000000]

Outputs: step, mean_reward, best_reward, fps (tab-separated)
Binned by --bin (default 1M steps) to keep output manageable.
"""
import re
import sys
from pathlib import Path


def parse_log(path: Path):
    pat = re.compile(
        r"Agent Steps: (\d+)M \| FPS: ([\d.]+) \| Last FPS: ([\d.]+) \| "
        r"(?:Collect Time: ([\d.]+) min \| Train RL Time: ([\d.]+) min \| )?"
        r"Mean Rewards: ([-\d.]+) \| Current Best: ([-\d.]+)"
    )
    rows = []
    for m in pat.finditer(path.read_text()):
        step = int(m.group(1)) * 1_000_000
        fps = float(m.group(2))
        mean = float(m.group(6))
        best = float(m.group(7))
        rows.append((step, mean, best, fps))
    return rows


def main():
    if len(sys.argv) < 2:
        print("usage: plot_training.py <log> [--bin N]")
        sys.exit(1)
    path = Path(sys.argv[1])
    bin_size = 1_000_000
    if "--bin" in sys.argv:
        i = sys.argv.index("--bin")
        bin_size = int(sys.argv[i + 1])

    rows = parse_log(path)
    if not rows:
        print("No data parsed.")
        return

    # Bin by step
    binned = {}
    for step, mean, best, fps in rows:
        b = (step // bin_size) * bin_size
        if b not in binned:
            binned[b] = []
        binned[b].append((mean, best, fps))

    print("step\tmean_reward\tbest_reward\tfps")
    for b in sorted(binned):
        data = binned[b]
        avg_mean = sum(d[0] for d in data) / len(data)
        max_best = max(d[1] for d in data)
        avg_fps = sum(d[2] for d in data) / len(data)
        print(f"{b}\t{avg_mean:.2f}\t{max_best:.2f}\t{avg_fps:.0f}")


if __name__ == "__main__":
    main()
