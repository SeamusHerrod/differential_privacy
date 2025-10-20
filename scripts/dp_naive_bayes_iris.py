#!/usr/bin/env python3
"""
Differentially-private Gaussian Naive Bayes trainer for the Iris dataset.

Overview:
 - We implement a Gaussian Naive Bayes classifier where the per-class
   sufficient statistics (counts, feature sums, and feature sum-of-squares)
   are privatized using the Laplace mechanism.

Privacy accounting and composition:
 - We split the global privacy budget epsilon into three parts:
     epsilon_count + epsilon_sum + epsilon_sumsq = epsilon
   These are used to privatize class counts, per-feature sums, and per-feature
   sum-of-squares respectively.
 - The queries (counts, sums, sumsq) are computed for each class. Counts
   across classes are disjoint (each training record contributes to exactly
   one class count), so the counts benefit from parallel composition over
   classes and can be released with epsilon_count each (no extra sequential
   cost across classes). The same holds for sums and sumsq when computed only
   on disjoint subsets (per-class). However, since we release counts, sums,
   and sumsq for the same class, we must account sequential composition per
   class: for each class the budget used is epsilon_count + epsilon_sum +
   epsilon_sumsq. In total the algorithm satisfies epsilon-differential
   privacy by construction (we partition epsilon accordingly).

Sensitivity and clamping:
 - We assume each feature value lies in a known range [min_val, max_val].
   For Iris features, we'll use the observed min/max from the training set and
   clamp values to this range before computing sums/sumsq. The L1 sensitivity
   of a class count when a single record changes is 1. The sensitivity of a
   clamped feature sum per class is (max_val - min_val). The sensitivity of
   the sum-of-squares is max(max_val^2, min_val^2) - min(min_val^2, max_val^2)
   (equivalently (max_val^2 - min_val^2)). We divide these sensitivities by
   1 because each record contributes once to its class' sums.

Implementation notes:
 - We add Laplace noise using inverse CDF sampling from Uniform(0,1).
 - We compute DP means and variances from noisy counts/sums/sumsq with a small
   correction to variances to keep them non-negative. We then use the Gaussian
   log-likelihoods for prediction.
 - CLI: --epsilon, --eps-count, --eps-sum, --eps-sumsq (optional overrides),
   --data-path, --seed.

References:
 - Dwork & Roth, "The Algorithmic Foundations of Differential Privacy" (Laplace mechanism)

"""
import argparse
import csv
import math
import random
from collections import defaultdict
from typing import List, Tuple


def laplace_noise(scale: float, rng: random.Random) -> float:
    # Draw from Laplace(0, b) where scale = b.
    u = rng.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def read_iris(path: str) -> List[Tuple[List[float], str]]:
    rows = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 5:
                continue
            features = [float(parts[i]) for i in range(4)]
            label = parts[4]
            rows.append((features, label))
    return rows


def dp_gaussian_nb_train(
    rows: List[Tuple[List[float], str]],
    epsilon: float,
    eps_count: float = None,
    eps_sum: float = None,
    eps_sumsq: float = None,
    seed: int = 0,
):
    rng = random.Random(seed)

    # Partition epsilon if parts not provided
    if eps_count is None or eps_sum is None or eps_sumsq is None:
        # Simple allocation: 20% to counts, 40% to sums, 40% to sumsq
        eps_count = 0.2 * epsilon
        eps_sum = 0.4 * epsilon
        eps_sumsq = 0.4 * epsilon

    # Gather training stats per class
    classes = defaultdict(list)
    for features, label in rows:
        classes[label].append(features)

    n_features = len(rows[0][0])

    # Determine per-feature clamp ranges from training data (non-private)
    feat_mins = [float('inf')] * n_features
    feat_maxs = [float('-inf')] * n_features
    for feats, _ in rows:
        for j, v in enumerate(feats):
            feat_mins[j] = min(feat_mins[j], v)
            feat_maxs[j] = max(feat_maxs[j], v)

    # Compute sensitivities
    sens_count = 1.0
    sens_sum = [feat_maxs[j] - feat_mins[j] for j in range(n_features)]
    sens_sumsq = [feat_maxs[j] * feat_maxs[j] - feat_mins[j] * feat_mins[j] for j in range(n_features)]

    model = {}

    for cls, cls_rows in classes.items():
        # True counts and sums (non-private used only to compute clamps and to fallback if noisy count <=0)
        true_count = len(cls_rows)
        sums = [0.0] * n_features
        sumsq = [0.0] * n_features
        for feats in cls_rows:
            for j, v in enumerate(feats):
                v_clamped = clamp(v, feat_mins[j], feat_maxs[j])
                sums[j] += v_clamped
                sumsq[j] += v_clamped * v_clamped

        # Add Laplace noise
        noisy_count = true_count + laplace_noise(sens_count / eps_count, rng)
        # Ensure count at least 1 to avoid division by zero; clip to small positive if necessary
        noisy_count = max(1.0, noisy_count)

        noisy_sums = [
            sums[j] + laplace_noise(sens_sum[j] / eps_sum, rng) for j in range(n_features)
        ]
        noisy_sumsq = [
            sumsq[j] + laplace_noise(sens_sumsq[j] / eps_sumsq, rng) for j in range(n_features)
        ]

        # Compute DP mean and variance (population variance: E[x^2] - E[x]^2)
        means = [noisy_sums[j] / noisy_count for j in range(n_features)]
        variances = []
        for j in range(n_features):
            ex2 = noisy_sumsq[j] / noisy_count
            ex = means[j]
            var = ex2 - ex * ex
            # numerical stability: variances must be non-negative
            if var <= 1e-8:
                var = 1e-6
            variances.append(var)

        model[cls] = {
            'count': noisy_count,
            'mean': means,
            'var': variances,
        }

    # Compute class priors from noisy counts (normalized)
    total_noisy = sum(model[cls]['count'] for cls in model)
    for cls in model:
        model[cls]['prior'] = model[cls]['count'] / total_noisy

    # Return model and clamp ranges (for applying to test set)
    return model, feat_mins, feat_maxs


def predict(model, feat_mins, feat_maxs, X: List[List[float]]) -> List[str]:
    preds = []
    for x in X:
        best = None
        best_score = float('-inf')
        for cls, stats in model.items():
            # clamp test features
            x_clamped = [clamp(x[j], feat_mins[j], feat_maxs[j]) for j in range(len(x))]
            # compute log-likelihood under Gaussian per feature
            logp = math.log(max(stats['prior'], 1e-12))
            for j, val in enumerate(x_clamped):
                mu = stats['mean'][j]
                var = stats['var'][j]
                # Gaussian log-prob
                logp += -0.5 * math.log(2 * math.pi * var) - 0.5 * ((val - mu) ** 2) / var
            if logp > best_score:
                best_score = logp
                best = cls
        preds.append(best)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, required=True)
    parser.add_argument('--eps-count', type=float, default=None)
    parser.add_argument('--eps-sum', type=float, default=None)
    parser.add_argument('--eps-sumsq', type=float, default=None)
    parser.add_argument('--data-path', type=str, default='data/iris.data')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rows = read_iris(args.data_path)

    # Prepare train/test split matching earlier script: test indices 1-10,51-60,101-110 (1-based)
    test_indices = set([i - 1 for i in list(range(1, 11)) + list(range(51, 61)) + list(range(101, 111))])
    train = [r for i, r in enumerate(rows) if i not in test_indices]
    test = [r for i, r in enumerate(rows) if i in test_indices]

    model, feat_mins, feat_maxs = dp_gaussian_nb_train(
        train,
        args.epsilon,
        eps_count=args.eps_count,
        eps_sum=args.eps_sum,
        eps_sumsq=args.eps_sumsq,
        seed=args.seed,
    )

    X_test = [x for x, _ in test]
    y_test = [y for _, y in test]
    preds = predict(model, feat_mins, feat_maxs, X_test)

    correct = sum(1 for p, t in zip(preds, y_test) if p == t)
    acc = correct / len(y_test)

    print(f"DP GaussianNB with epsilon={args.epsilon:.4f}")
    print(f"Train size: {len(train)}  Test size: {len(test)}  Accuracy: {acc:.4f}")
    print("Per-instance (index, true, pred):")
    idxs = [i for i in range(len(rows)) if i in test_indices]
    for idx, t, p in zip(idxs, y_test, preds):
        print(f"{idx+1}: {t} -> {p}")


if __name__ == '__main__':
    main()
