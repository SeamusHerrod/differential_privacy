import os
import math
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_DIR = os.path.normpath(DATA_DIR)
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# Configuration: files to load
groups = [
    ('original', 'adult.data'),
    ('minus_oldest', 'adult_minus_oldest.data'),
    ('minus_age26', 'adult_minus_age26.data'),
    ('minus_youngest', 'adult_minus_youngest.data'),
]

noisy_files = {
    0.5: {
        'original': os.path.join(DATA_DIR, 'noisy_results_eps0.500000_original.txt'),
        'minus_oldest': os.path.join(DATA_DIR, 'noisy_results_eps0.500000_minus_oldest.txt'),
        'minus_age26': os.path.join(DATA_DIR, 'noisy_results_eps0.500000_minus_age26.txt'),
        'minus_youngest': os.path.join(DATA_DIR, 'noisy_results_eps0.500000_minus_youngest.txt'),
    },
    1.0: {
        'original': os.path.join(DATA_DIR, 'noisy_results_eps1.000000_original.txt'),
        'minus_oldest': os.path.join(DATA_DIR, 'noisy_results_eps1.000000_minus_oldest.txt'),
        'minus_age26': os.path.join(DATA_DIR, 'noisy_results_eps1.000000_minus_age26.txt'),
        'minus_youngest': os.path.join(DATA_DIR, 'noisy_results_eps1.000000_minus_youngest.txt'),
    }
}

# Helpers

def read_noisy(path):
    vals = []
    with open(path) as f:
        for l in f:
            try:
                vals.append(float(l.strip()))
            except:
                pass
    return vals

def true_avg(path):
    vals = []
    with open(path) as f:
        for l in f:
            t = l.strip()
            if not t: continue
            tok = t.split(',')[0].strip()
            try:
                age = int(tok)
                if age > 25:
                    vals.append(age)
            except:
                continue
    return sum(vals) / len(vals)

# Load true averages
true_avgs = {}
for key, fname in groups:
    true_avgs[key] = true_avg(os.path.join(DATA_DIR, fname))

# Load noisy values and compute errors
errors = {eps: {} for eps in noisy_files}
mae = {eps: {} for eps in noisy_files}
for eps, mapping in noisy_files.items():
    for key in mapping:
        path = mapping[key]
        vals = read_noisy(path)
        if len(vals) == 0:
            errors[eps][key] = []
            mae[eps][key] = None
            continue
        ta = true_avgs[key]
        errs = [abs(v - ta) for v in vals]
        errors[eps][key] = errs
        mae[eps][key] = statistics.mean(errs)

# Plot 1: overlayed histograms per dataset (4 subplots)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()
colors = {0.5: '#1f77b4', 1.0: '#ff7f0e'}

for i, (key, _) in enumerate(groups):
    ax = axes[i]
    # combine to get common bins
    all_errs = []
    for eps in errors:
        all_errs.extend(errors[eps].get(key, []))
    if len(all_errs) == 0:
        continue
    max_err = max(all_errs)
    bins = min(50, max(5, int(max_err / 0.0005)))
    for eps in sorted(errors.keys()):
        errs = errors[eps].get(key, [])
        ax.hist(errs, bins=bins, alpha=0.6, density=True, color=colors[eps], label=f'ε={eps}')
    ax.set_title(key)
    ax.set_xlabel('Absolute error')
    ax.set_ylabel('Density')
    ax.legend()

plt.suptitle('Error histograms (absolute error) by dataset and ε')
out_hist = os.path.join(OUT_DIR, 'error_histograms.png')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(out_hist)
plt.close(fig)

# Plot 2: MAE vs epsilon for each dataset
fig2, ax2 = plt.subplots(figsize=(8,6))
eps_vals = sorted(noisy_files.keys())
for key, _ in groups:
    y = [mae[eps].get(key, float('nan')) for eps in eps_vals]
    ax2.plot(eps_vals, y, marker='o', label=key)

ax2.set_xlabel('ε')
ax2.set_ylabel('MAE (mean absolute error)')
ax2.set_title('MAE vs ε for each dataset')
ax2.legend()
ax2.grid(True)
out_mae = os.path.join(OUT_DIR, 'mae_vs_epsilon.png')
plt.savefig(out_mae)
plt.close(fig2)

# Small CSV summary
csv_out = os.path.join(OUT_DIR, 'mae_summary.csv')
with open(csv_out, 'w') as f:
    f.write('dataset,eps,mae\n')
    for eps in sorted(mae.keys()):
        for key in mae[eps]:
            f.write(f'{key},{eps},{mae[eps][key]}\n')

print('Wrote:', out_hist, out_mae, csv_out)
