"""Exact finite-population intervals for the uniform random product control only."""
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import hypergeom

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/product_replay'
summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
assert json.loads((OUT / 'verification.json').read_text())['passed']
rows = []
for seed in range(100, 105):
    base = OUT / f'stratified_random_from_zero192_seed{seed}'
    report = json.loads(base.with_suffix('.json').read_text())
    with np.load(base.with_suffix('.npz')) as a:
        y, domains, labeled = a['truth'], a['domains'], a['labeled']
        holdout = np.concatenate([a['validation_indices'], a['test_indices']])
        train = np.setdiff1d(np.arange(len(y)), holdout)
        with np.load(base.with_name(base.name + '_states.npz')) as states:
            mu, sigma = states['192_mu'], states['192_sigma']
    for domain in sorted(set(domains)):
        population = train[domains[train] == domain]
        observed = labeled[domains[labeled] == domain]
        N, n, k = len(population), len(observed), int(np.sum(y[observed] <= .05))
        assert N == 60 and n == 32
        possible = np.arange(k, N - n + k + 1)
        accepted = possible[(hypergeom.sf(k - 1, N, possible, n) >= .025) &
                            (hypergeom.cdf(k, N, possible, n) >= .025)]
        lo, hi = float(accepted.min() / N), float(accepted.max() / N)
        actual = float(np.mean(y[population] <= .05))
        unobserved = np.setdiff1d(population, observed)
        p = ndtr((.05 - mu) / np.maximum(sigma, 1e-12))
        gp = float(np.mean(p[population]))
        gp_with_known = float((k + p[unobserved].sum()) / N)
        rows.append(dict(seed=seed, domain=domain, N=N, n=n, successes=k,
            estimate=float(k / n), exact_interval95=[lo, hi], actual_train_prevalence=actual,
            interval_contains_true_train_prevalence=lo <= actual <= hi,
            sampling_error=abs(k / n - actual), gp_error=abs(gp - actual),
            gp_with_observed_labels_error=abs(gp_with_known - actual)))
result = dict(job_id=summary['protocol']['job_id'], passed=True, cases=30,
    estimand='Finite fixed train population (60 texts per domain), not test or all 100 domain texts.',
    design='Simple random sampling without replacement within each train domain: n=32 of N=60.',
    interval='Conservative equal-tail 95% confidence set by inversion of the exact hypergeometric distribution.',
    limitations=['No such unbiasedness or interval is claimed for cluster or adaptive sigma samples.',
                 'Coverage count on one finite pool is a diagnostic, not a new coverage guarantee.',
                 'GP entries are point-estimate controls; their sigma is not a design-based confidence interval.'],
    interval_coverage_count=sum(r['interval_contains_true_train_prevalence'] for r in rows),
    means={k: float(np.mean([r[k] for r in rows])) for k in ['sampling_error', 'gp_error', 'gp_with_observed_labels_error']},
    rows=rows)
(OUT / 'sampling_design.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
