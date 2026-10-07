"""Isolate independent-encoder text-span mismatch on frozen labels and splits."""
import json
import platform
import time

import numpy as np
import torch

import review_core as core
import review_surrogates as models

started = time.perf_counter()
assert torch.cuda.is_available(), 'This encoder experiment uses the authorized Colab GPU.'
free, total = torch.cuda.mem_get_info()
environment = dict(python=platform.python_version(), torch=torch.__version__,
    gpu=torch.cuda.get_device_name(), free_gib=free / 2**30, total_gib=total / 2**30)
print('Environment:', json.dumps(environment), flush=True)
datasets = ['controlled', 'natural_fold0', 'natural_fold1', 'natural_fold2']
out = 'prefix_control'
protocol = dict(job_id=GP_JOB_ID, environment=environment, datasets=datasets, seeds=list(range(100, 105)),
    budget=192, methods=['original_text', 'qwen_visible_prefix'], kernel='gp_matern32', target='raw exact JSD',
    hypothesis='The encoder seeing text beyond the measured Qwen prefix may damage prediction.',
    intervention='Crop text at the maximum cached consumed-token character end before independent MiniLM encoding.',
    kept_fixed='Texts/labels/row IDs/split/train groups/random labeled indices/pooling/encoder revision/kernel/target.',
    primary='Validation macro-domain theta absolute error at epsilon=0.05.',
    promotion_gate=dict(theta_relative_improvement=.10, improved_datasets_at_least=3,
                        maximum_dataset_theta_regression=.02, maximum_relative_mae_regression=.05),
    scope='Only a frozen-label pooled GP control; no adaptive selection or token-model rerun.',
    limitations=['Previously inspected research pools; seed SD is algorithmic randomness.',
                 'Independent encoder retains its own 256-token cap and may omit part of a Qwen prefix.',
                 'Qwen offsets are preprocessing boundaries, not Qwen activations.',
                 'This experiment does not isolate the choice of encoder architecture.'], physical_qwen_measurements=0)
core.save_json(out + '/protocol.json', protocol)
files, reports, span_reports, regressions = [out + '/protocol.json'], [], [], []


def macro(metrics):
    return float(np.mean([v['theta']['0.05']['absolute_error'] for d, v in metrics.items() if d != '__all__']))


for dataset in datasets:
    rows = core.load_rows('data/' + dataset + '.jsonl')
    with np.load(core.ROOT / 'frozen_inputs' / (dataset + '.npz')) as a:
        X, y, length, domains, groups = [a[k].copy() for k in ['X', 'y', 'length', 'domains', 'groups']]
        split = {part: a[part + '_indices'].copy() for part in ['train', 'validation', 'test']}
        row_ids = a['ids'].copy()
    assert row_ids.tolist() == [r['sample_id'] for r in rows]
    descriptors = json.loads((core.ROOT / 'inputs/visible_spans.json').read_text())
    spans = {r['sample_id']: r for r in descriptors if r['dataset'] == dataset}
    cropped = []
    for row in rows:
        endpoint = spans[row['sample_id']]['qwen_end']
        assert 0 < endpoint <= len(row['text'])
        value = dict(row, text=row['text'][:endpoint])
        value['text_hash'] = core.fingerprint(value['text'])
        cropped.append(value)
    core.save_rows('data/' + dataset + '_prefix.jsonl', cropped)
    path = models.embeddings(dataset + '_prefix')
    with np.load(path) as a:
        prefix_X = a['X'].astype(float)
        encoder_lengths = a['original_encoder_lengths'].copy()
    assert len(prefix_X) == len(X) and np.isfinite(prefix_X).all()
    span_reports.append(dict(dataset=dataset, n=len(rows), changed_texts=sum(a['text'] != b['text'] for a, b in zip(rows, cropped)),
        encoder_still_truncated=int(np.sum(encoder_lengths > core.CFG['encoder_max_length'])),
        feature_mean_l2_change=float(np.mean(np.linalg.norm(prefix_X - X, axis=1)))))
    files.extend([out + '/' + dataset + '_features.npz'])
    np.savez_compressed(core.ROOT / files[-1], X=prefix_X, ids=row_ids,
                        encoder_original_lengths=encoder_lengths,
                        qwen_character_ends=np.asarray([spans[r['sample_id']]['qwen_end'] for r in rows]))
    for seed in range(100, 105):
        labeled = models.labeled_order(rows, split, seed)[:192]
        for method, features in [('original_text', X), ('qwen_visible_prefix', prefix_X)]:
            fitted = models.fit_predictor('gp_matern32', features[labeled], y[labeled],
                        domains[labeled], length[labeled], seed, groups[labeled])
            report = dict(dataset=dataset, seed=seed, method=method, budget=192,
                          labeled=labeled.tolist(), kernel=fitted['kernel'], fit_seconds=fitted['seconds'],
                          warnings=fitted['warnings'], metrics={})
            arrays = dict(labeled=labeled, ids=row_ids, truth=y, domains=domains)
            for part in ['validation', 'test']:
                query = split[part]
                mu, sigma = models.predict(fitted, features[query], domains[query], length[query])
                report['metrics'][part] = models.quality(y[query], mu, domains[query], y[labeled], domains[labeled], sigma)
                arrays.update({part + '_indices': query, part + '_mu': mu, part + '_sigma': sigma})
                if method == 'original_text':
                    with np.load(core.ROOT / 'predictors' / f'{dataset}_seed{seed}.npz') as old:
                        error = max(float(np.max(abs(mu - old[f'gp_matern32_192_{part}_mu']))),
                                    float(np.max(abs(sigma - old[f'gp_matern32_192_{part}_sigma']))))
                        assert error < 1e-10
                        regressions.append(dict(dataset=dataset, seed=seed, part=part, max_error=error))
            base = f'{out}/{dataset}_{method}_seed{seed}'
            core.save_json(base + '.json', report)
            np.savez_compressed(core.ROOT / (base + '.npz'), **arrays)
            files.extend([base + '.json', base + '.npz']); reports.append(report)
            print(dataset, seed, method, 'test MAE:', round(report['metrics']['test']['__all__']['mae'], 6), flush=True)
aggregates = []
for dataset in datasets:
    for method in protocol['methods']:
        chosen = [r for r in reports if r['dataset'] == dataset and r['method'] == method]
        for part in ['validation', 'test']:
            maes = [r['metrics'][part]['__all__']['mae'] for r in chosen]
            thetas = [macro(r['metrics'][part]) for r in chosen]
            aggregates.append(dict(dataset=dataset, method=method, part=part, n=5,
                mae=float(np.mean(maes)), mae_seed_sd=float(np.std(maes, ddof=1)),
                theta_error=float(np.mean(thetas)), theta_seed_sd=float(np.std(thetas, ddof=1))))
gates = []
for dataset in datasets:
    baseline, trial = [next(r for r in aggregates if r['dataset'] == dataset and r['part'] == 'validation' and r['method'] == m)
                       for m in protocol['methods']]
    gates.append(dict(dataset=dataset, theta_improvement=1 - trial['theta_error'] / baseline['theta_error'],
        theta_regression=trial['theta_error'] - baseline['theta_error'], relative_mae_regression=trial['mae'] / baseline['mae'] - 1))
promoted = (sum(r['theta_improvement'] >= .10 for r in gates) >= 3 and
            all(r['theta_regression'] <= .02 and r['relative_mae_regression'] <= .05 for r in gates))
summary = dict(protocol=protocol, cases=len(reports), span_reports=span_reports, regressions=regressions,
               aggregates=aggregates, validation_gates=gates, candidate_for_product_check=promoted,
               seconds=time.perf_counter() - started)
core.save_json(out + '/summary.json', summary); files.append(out + '/summary.json')
print('Prefix control complete:', len(reports), 'cases;', 'promotion gate:', promoted, flush=True)
core.export(files, GP_JOB_ID)
