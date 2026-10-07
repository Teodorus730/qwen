"""Paired predictors and acquisition: external embeddings, hidden label oracle."""
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import oct_core as core


def feature_definition():
    pooling = core.CFG.get('encoder_pooling', 'mean')
    if pooling == 'mean':
        return 'last_hidden_attention_mask_mean_l2_standard_including_cls_sep_v1'
    if pooling == 'cls':
        return 'last_hidden_cls_l2_v1'
    raise ValueError('encoder_pooling: mean or cls')


def embeddings(name):
    import torch
    rows = core.load_rows('data/' + name + '.jsonl')
    path = core.ROOT / 'features' / (name + '.npz')
    identity = core.fingerprint({'ids': [r['sample_id'] for r in rows], 'texts': [r['text_hash'] for r in rows],
               'encoder': core.CFG['models']['encoder'], 'max_length': core.CFG['encoder_max_length'],
               'feature': feature_definition()})
    if path.exists():
        with np.load(path) as saved: assert str(saved['identity']) == identity
        return path
    tok, model = core.tokenizer('encoder'), core.model('encoder')
    vectors, lengths = [], []
    started = time.perf_counter()
    for start in range(0, len(rows), 32):
        batch = rows[start:start+32]
        inputs = tok([r['text'] for r in batch], padding=True, truncation=True,
                     max_length=core.CFG['encoder_max_length'], return_tensors='pt').to('cuda')
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state.float()
            mask = inputs['attention_mask'].unsqueeze(-1)
            vector = hidden[:, 0] if core.CFG.get('encoder_pooling', 'mean') == 'cls' else (hidden*mask).sum(1)/mask.sum(1)
            vector = torch.nn.functional.normalize(vector, p=2, dim=1)
        vectors.append(vector.cpu().numpy())
        lengths.extend(inputs['attention_mask'].sum(1).cpu().tolist())
    path.parent.mkdir(exist_ok=True)
    original_lengths=[len(x) for x in tok([r['text'] for r in rows],truncation=False,verbose=False)['input_ids']]
    np.savez_compressed(path, X=np.concatenate(vectors), lengths=np.array(lengths), original_encoder_lengths=np.array(original_lengths), identity=np.array(identity),
                        ids=np.array([r['sample_id'] for r in rows]))
    core.save_json('features/' + name + '_metadata.json', {'identity': identity, 'n': len(rows),
                    'encoder': core.CFG['models']['encoder'], 'feature': feature_definition(),
                    'seconds': time.perf_counter()-started})
    return path


def split_data(rows, validation_per_domain=None, test_per_domain=None):
    # Group near-identical questions within each subject before holdout splitting.
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.neighbors import NearestNeighbors
    domains = np.array([r['domain'] for r in rows])
    probe_path=core.ROOT/'data/probe.jsonl'
    probe_ids={r['sample_id'] for r in core.load_rows('data/probe.jsonl')} if probe_path.exists() else set()
    parts = {'train': [], 'validation': [], 'test': []}
    groups = np.arange(len(rows))
    for domain in sorted(set(domains)):
        indices = np.flatnonzero(domains == domain)
        vectors = HashingVectorizer(analyzer='char', ngram_range=(5,5), n_features=2**18,
                      alternate_sign=False, dtype=np.float32).transform([rows[i]['text'] for i in indices])
        distances, neighbors = NearestNeighbors(n_neighbors=min(8,len(indices)), metric='cosine',
                               algorithm='brute').fit(vectors).kneighbors(vectors)
        parent = list(range(len(indices)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        for i in range(len(indices)):
            for distance, j in zip(distances[i], neighbors[i]):
                if distance <= 1-core.CFG['near_duplicate_cosine']:
                    a,b = find(i),find(int(j)); parent[max(a,b)] = min(a,b)
        grouped = {}
        for local, index in enumerate(indices):
            group = int(indices[find(local)])
            groups[index] = group; grouped.setdefault(group, []).append(int(index))
        ordered = sorted(grouped.values(), key=lambda ids: core.fingerprint(['split',core.CFG['data_seed'],rows[ids[0]]['sample_id']]))
        reserved=[ids for ids in ordered if any(rows[i]['sample_id'] in probe_ids for i in ids)]
        ordered=[ids for ids in ordered if not any(rows[i]['sample_id'] in probe_ids for i in ids)]
        for ids in reserved: parts['train'].extend(ids)
        remaining = list(ordered)
        for part in ['test','validation']:
            requested = test_per_domain if part=='test' else validation_per_domain
            target = max(20, round(.2*len(indices))) if requested is None else requested
            while len([i for i in parts[part] if domains[i]==domain]) < target:
                if not remaining:
                    raise ValueError(f'{domain}: holdouts exceed available document groups')
                parts[part].extend(remaining.pop(0))
        for group in remaining: parts['train'].extend(group)
    parts = {k: np.array(sorted(v), dtype=int) for k,v in parts.items()}
    assert not set(groups[parts['train']]) & set(groups[parts['test']])
    assert not set(groups[parts['train']]) & set(groups[parts['validation']])
    assert not set(groups[parts['test']]) & set(groups[parts['validation']])
    return parts, groups


def labeled_order(rows, split, seed):
    rng = np.random.default_rng(seed)
    domains = np.array([r['domain'] for r in rows])
    initial, remaining = [], []
    unique = sorted(set(domains))
    per_domain = 24//len(unique)
    for domain in unique:
        ids = split['train'][domains[split['train']]==domain].copy()
        rng.shuffle(ids)
        initial.extend(ids[:per_domain]); remaining.extend(ids[per_domain:])
    rng.shuffle(remaining)
    return np.array(initial+remaining,dtype=int)


def gp(kind, X, seed, optimize=True):
    # Matched kernel families: identical amplitude/noise/bounds, only shape changes.
    distances = np.sqrt(np.maximum(0,2-2*np.clip(X@X.T,-1,1)))
    median = float(np.median(distances[np.triu_indices(len(X),1)])) or 1.
    bounds = (.05*median,10*median)
    radial = RBF(median,bounds) if kind=='gp_rbf' else Matern(median,bounds,nu=1.5 if kind=='gp_matern32' else 2.5)
    kernel = ConstantKernel(1.,(.01,100.))*radial+WhiteKernel(.05,(1e-5,2.))
    return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed,
                 n_restarts_optimizer=0, alpha=1e-8, optimizer='fmin_l_bfgs_b' if optimize else None)


def quality(y, mu, domains, train_y, train_domains, sigma=None):
    out = {}
    for domain in ['__all__']+sorted(set(domains)):
        mask = np.ones(len(y),dtype=bool) if domain=='__all__' else domains==domain
        truth,pred = y[mask],mu[mask]
        baseline = train_y.mean() if domain=='__all__' else train_y[train_domains==domain].mean()
        rho = spearmanr(truth,pred).statistic if np.std(pred)>1e-12 and np.std(truth)>1e-12 else None
        report = {'n':int(mask.sum()),'mae':float(mean_absolute_error(truth,pred)),
                  'rmse':float(np.sqrt(mean_squared_error(truth,pred))), 'r2':float(r2_score(truth,pred)),
                  'spearman':None if rho is None else float(rho),
                  'std_ratio':float(np.std(pred)/max(np.std(truth),1e-12)),
                  'train_mean_mae':float(np.abs(truth-baseline).mean())}
        if sigma is not None:
            report['theta']={}
            for epsilon in [.03,.05,.07]:
                probabilities=ndtr((epsilon-pred)/np.maximum(sigma[mask],1e-12))
                empirical=float((truth<=epsilon).mean())
                report['theta'][str(epsilon)]={'predicted':float(probabilities.mean()),'empirical':empirical,
                          'absolute_error':float(abs(probabilities.mean()-empirical)),
                          'brier':float(np.mean((probabilities-(truth<=epsilon))**2))}
        out[domain]=report
    return out


def fit_predictor(kind, X, y, domains, length, seed, groups=None):
    started=time.perf_counter();warns=[]
    if kind=='global_mean':
        return {'kind':kind,'mean':float(y.mean()),'seconds':0.,'warnings':[]}
    if kind=='subject_mean':
        return {'kind':kind,'means':{d:float(y[domains==d].mean()) for d in sorted(set(domains))},'seconds':0.,'warnings':[]}
    candidates=[]
    if kind in ['ridge','length_only']:
        candidates=[make_pipeline(StandardScaler(),Ridge(alpha=a)) for a in [.1,1.,10.,100.]]
    elif kind=='bayesian_ridge':
        candidates=[make_pipeline(StandardScaler(),BayesianRidge())]
    elif kind=='knn':
        candidates=[KNeighborsRegressor(n_neighbors=k,weights='distance') for k in [3,7,15]]
    elif kind=='krr':
        candidates=[TransformedTargetRegressor(regressor=KernelRidge(alpha=a,kernel='rbf',gamma=g),
                    transformer=StandardScaler()) for a in [.01,.1,1.] for g in [.5,2.,8.]]
    elif kind=='extra_trees':
        candidates=[ExtraTreesRegressor(n_estimators=150,min_samples_leaf=k,max_features=.7,
                    random_state=seed,n_jobs=2) for k in [2,5,10]]
    elif kind.startswith('gp_'):
        candidates=[gp(kind,X,seed)]
    else:
        raise ValueError(kind)
    inputs=length.reshape(len(length),-1) if kind=='length_only' else X
    cv=KFold(3,shuffle=True,random_state=seed) if groups is None else GroupKFold(3,shuffle=True,random_state=seed)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        scores=[float(np.mean(cross_val_score(candidate,inputs,y,cv=cv,groups=groups,scoring='neg_mean_absolute_error',n_jobs=1)))
                for candidate in candidates] if len(candidates)>1 else [None]
        best=int(np.argmax(scores)) if len(candidates)>1 else 0
        estimator=clone(candidates[best]).fit(inputs,y)
        warns=[str(w.message) for w in caught]
    return {'kind':kind,'estimator':estimator,'seconds':time.perf_counter()-started,'warnings':warns,
            'cv_scores':scores,'selected_candidate':best,'parameters':str(estimator),
            'kernel':str(estimator.kernel_) if kind.startswith('gp_') else None}


def predict(predictor,X,domains,length):
    kind=predictor['kind']
    if kind=='global_mean': return np.full(len(X),predictor['mean']),None
    if kind=='subject_mean': return np.array([predictor['means'][d] for d in domains]),None
    inputs=length.reshape(len(length),-1) if kind=='length_only' else X
    estimator=predictor['estimator']
    with threadpool_limits(limits=1):
        if kind.startswith('gp_'):
            return estimator.predict(inputs,return_std=True)
        if kind=='bayesian_ridge':
            return estimator[-1].predict(estimator[0].transform(inputs),return_std=True)
        return estimator.predict(inputs),None


def load_data(name):
    rows=core.load_rows('data/'+name+'.jsonl')
    with np.load(core.ROOT/'features'/ (name+'.npz')) as data:
        X=data['X'].astype(float)
        original_encoder_lengths=data['original_encoder_lengths'].copy()
        assert data['ids'].tolist()==[r['sample_id'] for r in rows]
    values=core.measurement_table(rows)
    assert len(values)==len(rows),'Benchmark ground truth incomplete'
    byid={r['sample_id']:r for r in values}
    y=np.array([byid[r['sample_id']]['exact'] for r in rows])
    length=np.column_stack([[byid[r['sample_id']]['n_tokens'] for r in rows],
                           [len(r['text']) for r in rows],original_encoder_lengths]).astype(float)
    domains=np.array([r['domain'] for r in rows])
    split,groups=split_data(rows)
    return rows,X,y,length,domains,split,groups


def predictor_benchmark(name, seed):
    rows,X,y,length,domains,split,groups=load_data(name)
    order=labeled_order(rows,split,seed)
    reports=[];arrays={}
    for budget in core.CFG['budgets']:
        labeled=order[:budget]
        assert len(labeled)==budget
        for kind in ['global_mean','subject_mean','length_only','ridge','bayesian_ridge','knn','krr','extra_trees',
                     'gp_rbf','gp_matern32','gp_matern52']:
            fitted=fit_predictor(kind,X[labeled],y[labeled],domains[labeled],length[labeled],seed,groups[labeled])
            report={k:v for k,v in fitted.items() if k!='estimator'}
            report.update(dataset=name,seed=seed,budget=budget,labeled=labeled.tolist(),metrics={})
            for part in ['validation','test']:
                ids=split[part]
                mu,sigma=predict(fitted,X[ids],domains[ids],length[ids])
                report['metrics'][part]=quality(y[ids],mu,domains[ids],y[labeled],domains[labeled],sigma)
                arrays[f'{kind}_{budget}_{part}_mu']=mu
                if sigma is not None: arrays[f'{kind}_{budget}_{part}_sigma']=sigma
            reports.append(report)
    relative=f'predictors/{name}_seed{seed}'
    core.save_json(relative+'.json',reports)
    path=core.ROOT/(relative+'.npz')
    np.savez_compressed(path,**arrays)
    core.save_json('splits/'+name+'.json',{'parts':{k:v.tolist() for k,v in split.items()},'near_duplicate_groups':groups.tolist()})
    return relative


def acquisition_benchmark(name,seed,kind='gp_matern32',budget=96):
    rows,X,y,length,domains,split,groups=load_data(name)
    initial=labeled_order(rows,split,seed)[:24]
    candidates=split['train']
    rank=sorted(candidates,key=lambda i:(-y[i],rows[i]['sample_id']))
    top10,top20=set(rank[:10]),set(rank[:20])
    reports=[];predictions={}
    for policy in ['random','stratified_random','sigma','ucb5']:
        rng=np.random.default_rng(seed)
        labeled=initial.copy();events=[];history=[]
        while True:
            fitted=fit_predictor(kind,X[labeled],y[labeled],domains[labeled],length[labeled],seed,groups[labeled])
            unseen=np.setdiff1d(candidates,labeled)
            history.append({'n':len(labeled),'best':float(y[labeled].max()),
                            'recall10':len(set(labeled)&top10)/10,'recall20':len(set(labeled)&top20)/20,
                            'top_discovered_mean':float(np.sort(y[labeled])[-10:].mean())})
            if len(labeled)==budget: break
            count=min(12,budget-len(labeled))
            if policy=='random': chosen=rng.choice(unseen,count,replace=False)
            elif policy=='stratified_random':
                chosen=[]
                for _ in range(count):
                    available=np.setdiff1d(unseen,chosen)
                    possible=sorted(set(domains[available]))
                    domain=min(possible,key=lambda d:(sum(domains[labeled]==d)+sum(domains[chosen]==d),d))
                    chosen.append(int(rng.choice(available[domains[available]==domain])))
                chosen=np.array(chosen)
            else:
                mu,sigma=predict(fitted,X[unseen],domains[unseen],length[unseen])
                score=sigma if policy=='sigma' else mu+5*sigma
                chosen=unseen[np.argsort(-score,kind='stable')[:count]]
                events.extend({'index':int(i),'mu':float(mu[np.where(unseen==i)[0][0]]),
                     'sigma':float(sigma[np.where(unseen==i)[0][0]]),'before':len(labeled)} for i in chosen)
            labeled=np.concatenate([labeled,chosen])
        report={'dataset':name,'seed':seed,'kernel':kind,'policy':policy,'budget':budget,
                'initial':initial.tolist(),'labeled':labeled.tolist(),'history':history,'events':events,
                'recall10':history[-1]['recall10'],'recall20':history[-1]['recall20'],
                'discovery_auc':float(np.trapezoid([h['recall10'] for h in history],[h['n'] for h in history])/(budget-24)),
                'metrics':{},'warnings':fitted['warnings']}
        for part in ['validation','test']:
            ids=split[part];mu,sigma=predict(fitted,X[ids],domains[ids],length[ids])
            report['metrics'][part]=quality(y[ids],mu,domains[ids],y[labeled],domains[labeled],sigma)
            predictions[policy+'_'+part+'_mu']=mu;predictions[policy+'_'+part+'_sigma']=sigma
        reports.append(report)
    relative=f'acquisition/{name}_seed{seed}'
    core.save_json(relative+'.json',reports)
    np.savez_compressed(core.ROOT/(relative+'.npz'),**predictions)
    return relative
