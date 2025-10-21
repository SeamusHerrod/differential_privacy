#!/usr/bin/env python3

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
    # Parse per-instance true/pred lines like: "1: Iris-setosa -> Iris-virginica"
    pairs = []
    for line in out.splitlines():
        pm = re.match(r'^(\d+):\s*(\S+)\s*->\s*(\S+)', line)
        if pm:
            idx = int(pm.group(1))
            true = pm.group(2)
            pred = pm.group(3)
            pairs.append((true, pred))

    # Compute macro-precision and macro-recall from pairs if available
    macro_prec = None
    macro_rec = None
    if pairs:
        labels = sorted({t for t, p in pairs} | {p for t, p in pairs})
        tp = {l: 0 for l in labels}
        fp = {l: 0 for l in labels}
        fn = {l: 0 for l in labels}
        for t, p in pairs:
            if t == p:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1
        precs = []
        recs = []
        for l in labels:
            prec = tp[l] / (tp[l] + fp[l]) if (tp[l] + fp[l]) > 0 else 0.0
            rec = tp[l] / (tp[l] + fn[l]) if (tp[l] + fn[l]) > 0 else 0.0
            precs.append(prec)
            recs.append(rec)
        macro_prec = sum(precs) / len(precs)
        macro_rec = sum(recs) / len(recs)

    return acc, macro_prec, macro_rec, out


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
            acc, prec, rec, out = run_trial(python_exec, script_path, eps, seed)
            prec_str = f"{prec:.4f}" if prec is not None else 'NA'
            rec_str = f"{rec:.4f}" if rec is not None else 'NA'
            print(f"  trial {t+1}/{args.trials}: acc={acc:.4f} prec={prec_str} rec={rec_str}")
            rows.append({'epsilon': eps, 'trial': t, 'seed': seed, 'accuracy': acc, 'precision': prec if prec is not None else '', 'recall': rec if rec is not None else ''})

    # write CSV
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epsilon', 'trial', 'seed', 'accuracy', 'precision', 'recall'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # print summary per epsilon
    print('\nSummary:')
    eps_map = {}
    for r in rows:
        eps_map.setdefault(r['epsilon'], []).append(r['accuracy'])
    # compute summaries for accuracy and recall
    for eps in sorted(eps_map.keys()):
        vals = eps_map[eps]
        mean = statistics.mean(vals)
        stdev = statistics.pstdev(vals)
        # collect recalls if present
        recs = [r['recall'] for r in rows if r['epsilon'] == eps and r['recall'] != '']
        precs = [r['precision'] for r in rows if r['epsilon'] == eps and r['precision'] != '']
        if recs:
            mean_rec = statistics.mean(recs)
            std_rec = statistics.pstdev(recs)
        else:
            mean_rec = None
            std_rec = None
        if precs:
            mean_prec = statistics.mean(precs)
            std_prec = statistics.pstdev(precs)
        else:
            mean_prec = None
            std_prec = None
        if mean_rec is not None:
            print(f"epsilon={eps}: mean_acc={mean:.4f}, std={stdev:.4f} (n={len(vals)})  mean_rec={mean_rec:.4f}, std_rec={std_rec:.4f}")
        else:
            print(f"epsilon={eps}: mean_acc={mean:.4f}, std={stdev:.4f} (n={len(vals)})")

    print(f"Wrote results to {args.out_csv}")


if __name__ == '__main__':
    main()
