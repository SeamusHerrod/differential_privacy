#!/usr/bin/env python3
"""
Run DP GaussianNB experiments across multiple epsilons and record accuracies.

This script calls `scripts/dp_naive_bayes_iris.py` repeatedly and parses its
stdout to extract the reported accuracy. It writes `outputs/dp_nb_results.csv`
with columns: epsilon,trial,seed,accuracy and prints a small summary.
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import statistics


def run_trial(python_exec, script_path, epsilon, seed):
    cmd = [python_exec, script_path, '--epsilon', str(epsilon), '--seed', str(seed)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nStderr:\n{proc.stderr}")
    out = proc.stdout
    # Find the Accuracy line
    m = re.search(r'Accuracy:\s*([0-9]*\.?[0-9]+)', out)
    if not m:
        raise RuntimeError(f"Couldn't parse accuracy from output:\n{out}")
    acc = float(m.group(1))
    return acc, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilons', nargs='+', type=float, default=[0.5, 1, 2, 4, 8, 16])
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed-base', type=int, default=0)
    parser.add_argument('--dp-script', type=str, default='scripts/dp_naive_bayes_iris.py')
    parser.add_argument('--out-csv', type=str, default='outputs/dp_nb_results.csv')
    args = parser.parse_args()

    python_exec = sys.executable
    script_path = os.path.abspath(args.dp_script)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    for eps in args.epsilons:
        print(f"Running epsilon={eps} ({args.trials} trials)")
        for t in range(args.trials):
            seed = args.seed_base + t
            acc, out = run_trial(python_exec, script_path, eps, seed)
            print(f"  trial {t+1}/{args.trials}: acc={acc:.4f}")
            rows.append({'epsilon': eps, 'trial': t, 'seed': seed, 'accuracy': acc})

    # write CSV
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epsilon', 'trial', 'seed', 'accuracy'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # print summary per epsilon
    print('\nSummary:')
    eps_map = {}
    for r in rows:
        eps_map.setdefault(r['epsilon'], []).append(r['accuracy'])
    for eps in sorted(eps_map.keys()):
        vals = eps_map[eps]
        mean = statistics.mean(vals)
        stdev = statistics.pstdev(vals)
        print(f"epsilon={eps}: mean_acc={mean:.4f}, std={stdev:.4f} (n={len(vals)})")

    print(f"Wrote results to {args.out_csv}")


if __name__ == '__main__':
    main()
