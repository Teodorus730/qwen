"""Document-separated token experiment using the independent encoder's last layer."""
import time
import warnings

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import oct_core as core
import oct_surrogates as surrogates

ALIGNMENT = 'qwen_character_overlap_minilm_last_layer_l2_pooled_fallback_v1'


def aligned_features(name):
    import torch
    rows = core.load_rows('data/' + name + '.jsonl')
    folder = core.ROOT / 'token_features' / name
    folder.mkdir(parents=True, exist_ok=True)
    tok, model = core.tokenizer('encoder'), core.model('encoder')
    counts = {'positions': 0, 'overlap_positions': 0, 'fallback_positions': 0, 'documents': len(rows)}
    started = time.perf_counter()
    identity = core.fingerprint([ALIGNMENT, core.CFG['models']['encoder'],
                                 core.CFG['encoder_max_length'], core.CFG['qwen_max_length']])
    for row in rows:
        path = folder / (row['sample_id'] + '.npz')
        with np.load(core.measurement_path(row)) as labels:
            target_offsets = labels['offsets'].copy()
            target_ids = labels['ids'].copy()
        row_identity = core.fingerprint([identity,row['text_hash'],target_ids.tolist(),target_offsets.tolist()])
        cached = False
        if path.exists():
            with np.load(path) as saved:
                cached = str(saved['identity']) == row_identity
                if cached:
                    count, coverage = len(saved['X']), int(saved['overlap'].sum())
        if not cached:
            inputs = tok(row['text'], truncation=True, max_length=core.CFG['encoder_max_length'],
                         return_offsets_mapping=True, return_tensors='pt')
            offsets = inputs.pop('offset_mapping')[0].numpy()
            with torch.inference_mode():
                hidden = model(**inputs.to('cuda')).last_hidden_state[0].float().cpu().numpy()
            # The same encoder/context/pooling as the pooled comparison. No Qwen activations.
            pooled = hidden.mean(axis=0)
            vectors, overlap = [], []
            for lo, hi in target_offsets:
                weights = np.maximum(0, np.minimum(offsets[:, 1], hi) - np.maximum(offsets[:, 0], lo))
                matched = bool(weights.sum() > 0)
                vector = np.average(hidden, axis=0, weights=weights) if matched else pooled
                vectors.append(vector / max(np.linalg.norm(vector), 1e-12))
                overlap.append(matched)
            np.savez_compressed(path, X=np.asarray(vectors, dtype=np.float32),
                                overlap=np.asarray(overlap), identity=np.array(row_identity))
            count, coverage = len(vectors), sum(overlap)
        counts['positions'] += count
        counts['overlap_positions'] += coverage
        counts['fallback_positions'] += count - coverage
    counts.update(identity=identity, alignment=ALIGNMENT, seconds=time.perf_counter()-started,
                  coverage=counts['overlap_positions']/counts['positions'],
                  special_tokens='zero-length special spans do not match; unmatched Qwen spans use pooled feature',
                  semantics='At position t the target compares next-token distributions after prefix through t; the encoder is bidirectional')
    core.save_json('token_features/' + name + '_metadata.json', counts)
    return counts


def load_tokens(name, ids):
    features, targets, docs = [], [], []
    rows = core.load_rows('data/' + name + '.jsonl')
    for i in ids:
        with np.load(core.ROOT/'token_features'/name/(rows[i]['sample_id']+'.npz')) as saved:
            features.append(saved['X'].astype(float))
            count = len(saved['X'])
        # Targets are always read from the current model-pair cache, never the encoder cache.
        with np.load(core.measurement_path(rows[i])) as saved:
            target = saved['exact'].astype(float)
            assert len(target) == count
            targets.append(target)
            docs.append(np.full(count, i, dtype=int))
    return np.concatenate(features), np.concatenate(targets), np.concatenate(docs)


def text_means(values, docs, ids):
    return np.asarray([values[docs == i].mean() for i in ids])


def weighted_ridge(X, y, docs, alpha):
    # Each measured document contributes weight one, regardless of its token count.
    _, inverse, counts = np.unique(docs, return_inverse=True, return_counts=True)
    weights = 1. / counts[inverse]
    scaler = StandardScaler().fit(X, sample_weight=weights)
    estimator = Ridge(alpha=alpha).fit(scaler.transform(X), y, sample_weight=weights)
    return scaler, estimator


def benchmark(name, seed, budgets=(96, 192)):
    rows, pooled, text_y, length, domains, split, groups = surrogates.load_data(name)
    order = surrogates.labeled_order(rows, split, seed)
    reports, arrays = [], {}
    for budget in budgets:
        labeled = order[:budget]
        X, y, docs = load_tokens(name, labeled)
        cv_groups = groups[docs]
        scores = []
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            for alpha in [.1, 1., 10., 100.]:
                errors = []
                for train, val in GroupKFold(3, shuffle=True, random_state=seed).split(X, y, cv_groups):
                    scaler, ridge = weighted_ridge(X[train], y[train], docs[train], alpha)
                    ids = np.unique(docs[val])
                    means = text_means(ridge.predict(scaler.transform(X[val])), docs[val], ids)
                    errors.append(float(np.abs(means - text_y[ids]).mean()))
                scores.append(float(np.mean(errors)))
            alpha = [.1, 1., 10., 100.][int(np.argmin(scores))]
            scaler, ridge = weighted_ridge(X, y, docs, alpha)
        ridge_seconds = time.perf_counter()-started
        # Uniform per-document sampling; label cost remains measured texts.
        rng = np.random.default_rng(seed)
        per_doc = max(1, 768//budget)
        token_indices = []
        for i in labeled:
            candidates = np.flatnonzero(docs == i)
            token_indices.extend(rng.choice(candidates, min(per_doc, len(candidates)), replace=False))
        token_indices = np.asarray(token_indices, dtype=int)
        gp_X, gp_y = X[token_indices], y[token_indices]
        started = time.perf_counter()
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            gp = surrogates.gp('gp_matern32', gp_X, seed).fit(gp_X, gp_y)
        gp_seconds = time.perf_counter()-started
        variants = [
            ('token_ridge', {'alpha': alpha, 'train_doc_cv_mae': scores, 'seconds': ridge_seconds}),
            ('token_gp_matern32', {'kernel': str(gp.kernel_), 'seconds': gp_seconds,
                'token_observations': len(token_indices), 'per_document_limit': per_doc,
                'warnings': [str(w.message) for w in caught],
                'selected_token_docs': docs[token_indices].tolist(), 'selected_token_rows': token_indices.tolist()})]
        for kind, details in variants:
            report = dict(kind=kind, dataset=name, seed=seed, budget=budget, labeled=labeled.tolist(),
                          metrics={}, alignment=ALIGNMENT, **details)
            for part in ['validation', 'test']:
                ids = split[part]
                query, _, query_docs = load_tokens(name, ids)
                with threadpool_limits(limits=1):
                    prediction = ridge.predict(scaler.transform(query)) if kind=='token_ridge' else gp.predict(query)
                mu = text_means(prediction, query_docs, ids)
                report['metrics'][part] = surrogates.quality(text_y[ids], mu, domains[ids],
                                                            text_y[labeled], domains[labeled])
                arrays[f'{kind}_{budget}_{part}_mu'] = mu
            reports.append(report)
    relative = f'token_predictors/{name}_seed{seed}'
    core.save_json(relative+'.json', reports)
    np.savez_compressed(core.ROOT/(relative+'.npz'), **arrays)
    return relative
