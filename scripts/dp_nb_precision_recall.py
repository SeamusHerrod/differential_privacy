#!/usr/bin/env python3
"""
Run dp_naive_bayes_iris.py for multiple epsilons and compute precision/recall
on the specified test indices (1-10,51-60,101-110). Outputs CSV and prints
per-epsilon metrics.
"""
import argparse
import csv
import os
import subprocess
import sys
import re
from collections import defaultdict
from typing import List


TEST_INDICES = [i - 1 for i in list(range(1, 11)) + list(range(51, 61)) + list(range(101, 111))]


def run_dp(python_exec: str, script_path: str, epsilon: float, seed: int):
    cmd = [python_exec, script_path, '--epsilon', str(epsilon), '--seed', str(seed)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nStderr:\n{proc.stderr}")
    out = proc.stdout
    # parse lines 
    preds = {}
    for line in out.splitlines():
        m = re.match(r'^(\d+):\s*(\S+)\s*->\s*(\S+)', line)
        if m:
            idx = int(m.group(1)) - 1
            true = m.group(2)
            pred = m.group(3)
            preds[idx] = (true, pred)
    missing = [i for i in TEST_INDICES if i not in preds]
    if missing:
        raise RuntimeError(f"Missing predictions for indices: {missing}\nOutput:\n{out}")
    # return list of (true, pred) ordered by TEST_INDICES
    return [preds[i] for i in TEST_INDICES]


def precision_recall_from_pairs(pairs: List[tuple]):
    # pairs: list of (true_label, pred_label)
    labels = sorted({t for t, p in pairs} | {p for t, p in pairs})
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for t, p in pairs:
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    precision = {}
    recall = {}
    for l in labels:
        prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
        rec = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0
        precision[l] = prec
        recall[l] = rec
    # macro averages
    macro_prec = sum(precision.values()) / len(precision)
    macro_rec = sum(recall.values()) / len(recall)
    return precision, recall, macro_prec, macro_rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilons', nargs='+', type=float, default=[0.5, 1, 2, 4, 8, 16])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dp-script', type=str, default='scripts/dp_naive_bayes_iris.py')
    parser.add_argument('--out-csv', type=str, default='outputs/dp_nb_pr.csv')
    args = parser.parse_args()

    python_exec = sys.executable
    script_path = os.path.abspath(args.dp_script)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    for eps in args.epsilons:
        pairs = run_dp(python_exec, script_path, eps, args.seed)
        precision, recall, macro_p, macro_r = precision_recall_from_pairs(pairs)
        # record per-class and macro
        for cls in sorted(precision.keys()):
            rows.append({'epsilon': eps, 'class': cls, 'precision': precision[cls], 'recall': recall[cls]})
        rows.append({'epsilon': eps, 'class': 'macro', 'precision': macro_p, 'recall': macro_r})
        # print brief summary
        print(f"epsilon={eps}: macro-precision={macro_p:.4f}, macro-recall={macro_r:.4f}")

    # write CSV
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epsilon', 'class', 'precision', 'recall'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote results to {args.out_csv}")


if __name__ == '__main__':
    main()
