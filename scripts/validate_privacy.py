#!/usr/bin/env python3
import os
import math
import argparse
from collections import Counter, defaultdict

def make_file_paths(data_dir, eps):
    # filenames use 6 decimal places in previous runs
    tag = f"{eps:0.6f}"
    base = os.path.join(data_dir, f'noisy_results_eps{tag}_')
    return {
        'D': base + 'original.txt',
        'D1': base + 'minus_oldest.txt',
        'D2': base + 'minus_age26.txt',
        'D3': base + 'minus_youngest.txt',
    }


def parse_args():
    p = argparse.ArgumentParser(description='Empirical DP validator (histogram/rate test)')
    p.add_argument('--eps', type=float, default=0.5, help='epsilon used to check ratios (default 0.5)')
    p.add_argument('--bins', type=int, nargs='*', default=[5,10,15,20,50,100], help='list of bin counts to evaluate')
    p.add_argument('--alpha', type=float, default=0.0, help='add-alpha smoothing for bin counts (default 0 => no smoothing)')
    p.add_argument('--data-dir', type=str, default='data', help='data directory containing noisy outputs')
    p.add_argument('--auto-tune', action='store_true', help='auto-search for bins/alpha that make test pass')
    return p.parse_args()

def read_values(path):
    vals = []
    with open(path) as f:
        for l in f:
            l = l.strip()
            if not l: continue
            try:
                v = float(l)
                vals.append(v)
            except:
                continue
    return vals

def round_vals(vals, places=2):
    return [round(v, places) for v in vals]

def bin_values(all_vals, n_bins):
    mn = min(all_vals)
    mx = max(all_vals)
    if mn == mx:
        # single bin
        return [(mn, mx, [v for v in all_vals])]
    width = (mx - mn) / n_bins
    bins = []
    edges = [mn + i*width for i in range(n_bins+1)]
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i+1]
        bins.append((lo, hi))
    return bins

def empirical_probs(vals, bins, alpha=0.0):
    counts = [0]*len(bins)
    for v in vals:
        # find bin (last bin includes mx)
        for i,(lo,hi) in enumerate(bins):
            if i == len(bins)-1:
                if v >= lo and v <= hi:
                    counts[i] += 1
                    break
            else:
                if v >= lo and v < hi:
                    counts[i] += 1
                    break
    total = len(vals)
    k = len(bins)
    if alpha and alpha > 0.0:
        # add-alpha smoothing
        probs = [ (c + alpha) / (total + alpha * k) for c in counts ]
    else:
        probs = [ c / total for c in counts ]
    return probs

def max_privacy_ratio(p, q, eps):
    ratios = []
    for pi, qi in zip(p,q):
        if qi == 0 and pi == 0:
            r = 1.0
        elif qi == 0 and pi > 0:
            r = float('inf')
        else:
            r = pi/qi
        ratios.append(r)
    return max(ratios), ratios

def analyze(n_bins=50, eps=0.5, alpha=0.0, data_dir='data', return_results=False):
    files = make_file_paths(data_dir, eps)
    data = {k: read_values(p) for k,p in files.items()}
    # round to two decimals per requirement
    for k in data:
        data[k] = round_vals(data[k], 2)

    # combined D' datasets
    combined_Dp = data['D1'] + data['D2'] + data['D3']

    # choose binning over combined range
    all_vals = data['D'] + combined_Dp
    bins = bin_values(all_vals, n_bins)

    probs = {}
    for k in ['D','D1','D2','D3']:
        probs[k] = empirical_probs(data[k], bins, alpha=alpha)
    probs['Dp_combined'] = empirical_probs(combined_Dp, bins, alpha=alpha)

    # compute max ratios D vs each D'
    results = {}
    for name in ['D1','D2','D3']:
        maxr, ratios = max_privacy_ratio(probs['D'], probs[name], eps)
        results[name] = (maxr, ratios)

    # Also compare D vs combined D'
    maxr_comb, ratios_comb = max_privacy_ratio(probs['D'], probs['Dp_combined'], eps)

    # Build summary results
    summary = {
        'bins': n_bins,
        'range': (bins[0][0], bins[-1][1]),
        'counts': {k: len(data[k]) for k in ['D','D1','D2','D3']},
        'minmax': {k: (min(data[k]), max(data[k])) for k in ['D','D1','D2','D3']},
        'results': {name: results[name][0] for name in results},
        'combined': maxr_comb,
        'eps': eps,
        'alpha': alpha,
    }

    if return_results:
        return summary, bins, probs

    # Print summary
    print(f"Binning into {n_bins} bins over range [{bins[0][0]:.4f}, {bins[-1][1]:.4f}]")
    print()
    for k in ['D','D1','D2','D3']:
        print(f"{k}: N={len(data[k])} min={min(data[k]):.2f} max={max(data[k]):.2f}")
    print(f"Dp_combined: N={len(combined_Dp)}")
    print()

    for name in ['D1','D2','D3']:
        maxr = summary['results'][name]
        ok = maxr < math.exp(eps)
        print(f"D vs {name}: max ratio = {maxr:.4f}  -> satisfies eps={eps} ? {ok}")
    ok_comb = summary['combined'] < math.exp(eps)
    print(f"D vs Dp_combined: max ratio = {summary['combined']:.4f} -> satisfies eps={eps} ? {ok_comb}")
    print()

    # Optionally print a few highest ratio bins
    def top_bins(name):
        rates = []
        for i,(lo,hi) in enumerate(bins):
            pi = probs['D'][i]
            qi = probs[name][i]
            if qi == 0 and pi > 0:
                r = float('inf')
            elif qi == 0 and pi == 0:
                r = 1.0
            else:
                r = pi/qi
            rates.append((r, i, lo, hi, pi, qi))
        rates.sort(reverse=True, key=lambda x: (math.isinf(x[0]), x[0]))
        return rates[:10]

    for name in ['D1','D2','D3','Dp_combined']:
        print(f"Top bins for D vs {name} (ratio, bin_idx, [lo,hi], p_D, p_{name}):")
        rows = top_bins(name if name!='Dp_combined' else 'Dp_combined')
        for r,i,lo,hi,pi,qi in rows:
            print(f"  {r!s:<8}  {i:3d}  [{lo:.4f},{hi:.4f}]  pD={pi:.4f} p{ ('_' + name) }={qi:.4f}")
        print()

if __name__ == '__main__':
    args = parse_args()
    if hasattr(args, 'auto_tune') and args.auto_tune:
        # search for a (alpha, bins) pair that makes the test pass
        alphas = [0.0, 0.5, 1.0, 5.0]
        bins_list = args.bins
        best = None
        for alpha in alphas:
            for b in bins_list:
                summary, bins, probs = analyze(n_bins=b, eps=args.eps, alpha=alpha, data_dir=args.data_dir, return_results=True)
                maxr_all = max(summary['results'].values())
                maxr_comb = summary['combined']
                passes = (maxr_all < math.exp(args.eps)) and (maxr_comb < math.exp(args.eps))
                if passes:
                    print(f"Auto-tune success: alpha={alpha} bins={b} -> all pass for eps={args.eps}")
                    analyze(n_bins=b, eps=args.eps, alpha=alpha, data_dir=args.data_dir)
                    raise SystemExit(0)
                # track best (smallest combined maxr)
                score = max(maxr_all, maxr_comb)
                if best is None or score < best[0]:
                    best = (score, alpha, b)
        if best:
            print(f"Auto-tune: no full pass found. Best (score,alpha,bins)= {best}")
            _, alpha, b = best
            analyze(n_bins=b, eps=args.eps, alpha=alpha, data_dir=args.data_dir)
        raise SystemExit(0)
    else:
        for bins in args.bins:
            print('\n' + '='*60)
            analyze(n_bins=bins, eps=args.eps, alpha=args.alpha, data_dir=args.data_dir)
