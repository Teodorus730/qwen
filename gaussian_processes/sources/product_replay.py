"""Matched acquisition control on the exact split of the live MMLU product."""
import io
import json
import platform
import time
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import ndtr

import review_core as core
import review_surrogates as models
import review_live as product


started = time.perf_counter()
environment = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    environment.update(gpu=torch.cuda.get_device_name(), free_gib=free / 2**30, total_gib=total / 2**30)
print('Environment:', json.dumps(environment), flush=True)
with np.load(core.ROOT / 'inputs/controlled.npz') as a:
    X, y, length, domains, groups = [a[k].copy() for k in ['X', 'y', 'length', 'domains', 'groups']]
    row_ids = a['ids'].copy()
old_split = json.loads((core.ROOT / 'inputs/product_split.json').read_text())
old_map = json.loads((core.ROOT / 'inputs/product_map.json').read_text())
split = {k: np.asarray(v, dtype=int) for k, v in old_split['parts'].items()}
assert groups.tolist() == old_split['groups']
for a, b in [('train', 'test'), ('train', 'validation'), ('validation', 'test')]:
    assert not set(split[a]) & set(split[b]) and not set(groups[split[a]]) & set(groups[split[b]])
OUT = 'product_replay'
protocol = dict(job_id=GP_JOB_ID, dataset='controlled', seeds=list(range(100, 105)),
    initial='Original KMeans 4 clusters x 5 nearest points per each of 6 domains = 120',
    budget=192, transform='raw', kernel='gp_matern32', environment=environment,
    methods=['cluster_only120', 'cluster_random12', 'cluster_sigma1', 'cluster_sigma4',
             'cluster_sigma12', 'stratified_random_from_zero192'],
    scope='Offline replay on the actual product split; zero new physical model-pair measurements.',
    primary='Validation macro-domain theta error at epsilon=0.05; worst-domain decision separately.',
    batch_promotion_gate=dict(theta_relative_improvement=.10, maximum_relative_mae_regression=.02,
                              worst_domain_hits_not_lower=True),
    notes=['This test pool has already been inspected.',
           'Mean and seed SD describe algorithmic randomness, not independent datasets.',
           'Random-from-zero is a separate equal-budget sampling control; it skips the cluster start.',
           'Empirical fraction is reported only for the random-from-zero control.',
           'GP fitting time is real; offline replay does not measure Qwen inference savings.'])
core.save_json(OUT + '/protocol.json', protocol)
files = [OUT + '/protocol.json']
reports = []
regression = None


def fit(labeled, seed):
    # Only acquired targets are passed to the predictor.
    return models.fit_predictor('gp_matern32', X[labeled], y[labeled], domains[labeled], length[labeled], seed, groups[labeled])


def evaluate(fitted, labeled, report):
    arrays = dict(labeled=labeled, truth=y, domains=domains, ids=row_ids)
    for part in ['validation', 'test']:
        ids = split[part]
        mu, sigma = models.predict(fitted, X[ids], domains[ids], length[ids])
        metrics = models.quality(y[ids], mu, domains[ids], y[labeled], domains[labeled], sigma)
        scores = {d: v['theta']['0.05']['predicted'] for d, v in metrics.items() if d != '__all__'}
        actual = {d: v['theta']['0.05']['empirical'] for d, v in metrics.items() if d != '__all__'}
        chosen = min(scores, key=scores.get)
        report['metrics'][part] = metrics
        report['decisions'][part] = dict(predicted_worst=chosen,
            actual_worst=[d for d, v in actual.items() if v == min(actual.values())],
            correct=actual[chosen] == min(actual.values()))
        arrays.update({part + '_indices': ids, part + '_mu': mu, part + '_sigma': sigma})
        if report['method'] == 'stratified_random_from_zero192':
            report.setdefault('empirical', {})[part] = {}
            for domain in scores:
                p = float((y[labeled[domains[labeled] == domain]] <= .05).mean())
                report['empirical'][part][domain] = dict(predicted=p, empirical=actual[domain],
                                                        absolute_error=abs(p - actual[domain]))
    return arrays


for seed in range(100, 105):
    cfg = dict(old_map['config'], seed=seed)
    task = SimpleNamespace(config=cfg, X=X, domains=domains, split=split)
    initial = product.DomainMap.initial_design(task)
    assert len(initial) == 120 and len(set(initial)) == 120
    if seed == 100:
        assert initial.tolist() == old_split['initial']
    for method, batch in [('cluster_only120', 12), ('cluster_random12', 12),
                          ('cluster_sigma1', 1), ('cluster_sigma4', 4), ('cluster_sigma12', 12),
                          ('stratified_random_from_zero192', 12)]:
        rng = np.random.default_rng(seed)
        budget = 120 if method == 'cluster_only120' else 192
        labeled = initial.copy()
        if method == 'stratified_random_from_zero192':
            labeled = np.concatenate([rng.choice(split['train'][domains[split['train']] == d], 32, replace=False)
                                      for d in sorted(set(domains))])
        history, events, state_arrays = [], [], {}
        fit_seconds = 0.
        while True:
            assert set(labeled) <= set(split['train'])
            fitted = fit(labeled, seed)
            fit_seconds += fitted['seconds']
            mu, sigma = models.predict(fitted, X, domains, length)
            n = len(labeled)
            state_arrays[f'{n}_mu'] = mu
            state_arrays[f'{n}_sigma'] = sigma
            state_arrays[f'{n}_labeled'] = labeled.copy()
            history.append(dict(n=n, fit_seconds=fitted['seconds'], kernel=fitted['kernel']))
            if n == budget:
                break
            candidates = np.setdiff1d(split['train'], labeled)
            count = min(batch, budget - n)
            if method == 'cluster_random12':
                chosen = rng.choice(candidates, count, replace=False)
            else:
                chosen = candidates[np.argsort(-sigma[candidates], kind='stable')[:count]]
            events.append(dict(before=n, chosen=chosen.tolist()))
            labeled = np.concatenate([labeled, chosen])
        report = dict(seed=seed, method=method, budget=budget, batch_size=batch,
                      initial=initial.tolist() if method != 'stratified_random_from_zero192' else [],
                      labeled=labeled.tolist(), history=history, events=events, fit_seconds=fit_seconds,
                      metrics={}, decisions={}, warnings=fitted['warnings'])
        arrays = evaluate(fitted, labeled, report)
        if seed == 100 and method == 'cluster_sigma12':
            with np.load(core.ROOT / 'inputs/product_predictions.npz') as a:
                regression = dict(labeled_identical=labeled.tolist() == old_map['labeled'],
                    mu_max_difference=float(np.max(abs(mu - a['mu']))),
                    sigma_max_difference=float(np.max(abs(sigma - a['sigma']))))
            assert regression['labeled_identical'] and regression['mu_max_difference'] < 1e-10 and regression['sigma_max_difference'] < 1e-10
        base = f'{OUT}/{method}_seed{seed}'
        core.save_json(base + '.json', report)
        np.savez_compressed(core.ROOT / (base + '.npz'), **arrays)
        np.savez_compressed(core.ROOT / (base + '_states.npz'), **state_arrays)
        files.extend([base + '.json', base + '.npz', base + '_states.npz'])
        reports.append(report)
        print(seed, method, 'MAE:', round(report['metrics']['test']['__all__']['mae'], 6),
              'fits:', len(history), 'fit_s:', round(fit_seconds, 2), flush=True)


def macro(metrics):
    return float(np.mean([v['theta']['0.05']['absolute_error'] for d, v in metrics.items() if d != '__all__']))


aggregates = []
for method in protocol['methods']:
    rows = [r for r in reports if r['method'] == method]
    for part in ['validation', 'test']:
        row = dict(method=method, part=part, n=5, budget=rows[0]['budget'],
                   theta_error=float(np.mean([macro(r['metrics'][part]) for r in rows])),
                   theta_seed_sd=float(np.std([macro(r['metrics'][part]) for r in rows], ddof=1)),
                   worst_correct=sum(r['decisions'][part]['correct'] for r in rows),
                   fit_seconds=float(np.mean([r['fit_seconds'] for r in rows])))
        for metric in ['mae', 'r2']:
            values = [r['metrics'][part]['__all__'][metric] for r in rows]
            row[metric] = float(np.mean(values)); row[metric + '_seed_sd'] = float(np.std(values, ddof=1))
        if method == 'stratified_random_from_zero192':
            row['empirical_theta_error'] = float(np.mean([np.mean([v['absolute_error'] for v in r['empirical'][part].values()]) for r in rows]))
        aggregates.append(row)
baseline = next(r for r in aggregates if r['method'] == 'cluster_sigma12' and r['part'] == 'validation')
gates = {}
for method in ['cluster_sigma1', 'cluster_sigma4']:
    trial = next(r for r in aggregates if r['method'] == method and r['part'] == 'validation')
    gates[method] = dict(theta_improves_10_percent=trial['theta_error'] <= .9 * baseline['theta_error'],
                        mae_not_worse_over_2_percent=trial['mae'] <= 1.02 * baseline['mae'],
                        worst_hits_not_lower=trial['worst_correct'] >= baseline['worst_correct'])
summary = dict(protocol=protocol, cases=len(reports), regression=regression, aggregates=aggregates,
               validation_gates=gates, seconds=time.perf_counter() - started)
core.save_json(OUT + '/summary.json', summary); files.append(OUT + '/summary.json')
print('Regression:', json.dumps(regression), flush=True)
print('Validation gates:', json.dumps(gates), flush=True)
core.export(files, GP_JOB_ID)
