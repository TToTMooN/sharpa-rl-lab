#!/usr/bin/env python3
"""Quick analysis of experiments.tsv — prints summary grouped by milestone."""

import csv
from pathlib import Path

TSV = Path(__file__).parent / "experiments.tsv"


def main():
    rows = list(csv.DictReader(TSV.open(), delimiter="\t"))
    if not rows:
        print("No experiments yet.")
        return

    print(f"Total experiments: {len(rows)}")
    print()

    # Group by milestone
    by_milestone = {}
    for r in rows:
        by_milestone.setdefault(r["milestone"], []).append(r)

    for ms, experiments in sorted(by_milestone.items()):
        print(f"=== Milestone {ms} ({len(experiments)} experiments) ===")
        print(f"  {'ID':<5}{'description':<25}{'best_reward':>12}{'final':>12}  status")
        for r in experiments:
            br = r["best_reward"] if r["best_reward"] != "-" else "—"
            fr = r["mean_reward_final"] if r["mean_reward_final"] != "-" else "—"
            print(f"  {r['id']:<5}{r['description']:<25}{br:>12}{fr:>12}  {r['status']}/{r['decision']}")
        print()

    # Highlight best per milestone
    print("=== Best per milestone (by best_reward) ===")
    for ms, experiments in sorted(by_milestone.items()):
        numeric = [e for e in experiments if e["best_reward"] not in ("-", "")]
        if not numeric:
            continue
        best = max(numeric, key=lambda e: float(e["best_reward"]))
        print(f"  {ms}: EXP-{best['id']} {best['description']} — {best['best_reward']}")


if __name__ == "__main__":
    main()
