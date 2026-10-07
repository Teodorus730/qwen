"""Last-priority hypothesis: train-only supervised PLS projection before GP."""
import time
import warnings

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import oct_core as core
import oct_surrogates as models


def fit(X, y, components, seed):
    projection = PLSRegression(n_components=components, scale=True, max_iter=500).fit(X,y)
    latent = projection.transform(X)
    median = float(np.median(pdist(latent))) or 1.
    kernel = ConstantKernel(1.,(.01,100.))*Matern(median,(.05*median,10*median),nu=1.5)+WhiteKernel(.05,(1e-5,2.))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8,
                                  n_restarts_optimizer=0,random_state=seed).fit(latent,y)
    return projection,gp


def benchmark(name,seed):
    rows,X,y,length,domains,split,groups=models.load_data(name)
    order=models.labeled_order(rows,split,seed)
    reports,arrays=[],{}
    for budget in core.CFG['budgets']:
        labeled=order[:budget]; scores=[]
        started=time.perf_counter()
        with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            cv=list(GroupKFold(3,shuffle=True,random_state=seed).split(X[labeled],y[labeled],groups[labeled]))
            for components in [2,4,8,16]:
                errors=[]
                for train,val in cv:
                    a,b=labeled[train],labeled[val]
                    projection,gp=fit(X[a],y[a],components,seed)
                    errors.append(float(np.abs(y[b]-gp.predict(projection.transform(X[b]))).mean()))
                scores.append(float(np.mean(errors)))
            components=[2,4,8,16][int(np.argmin(scores))]
            projection,gp=fit(X[labeled],y[labeled],components,seed)
        report=dict(kind='pls_gp_matern32',dataset=name,seed=seed,budget=budget,labeled=labeled.tolist(),
                    selected_components=components,train_group_cv_mae=scores,
                    kernel=str(gp.kernel_),seconds=time.perf_counter()-started,
                    warnings=[str(w.message) for w in caught],metrics={},
                    geometry='Euclidean PLS scores, train fitted scale; matched median-relative kernel bounds')
        for part in ['validation','test']:
            ids=split[part]
            with threadpool_limits(limits=1):
                mu,sigma=gp.predict(projection.transform(X[ids]),return_std=True)
            report['metrics'][part]=models.quality(y[ids],mu,domains[ids],y[labeled],domains[labeled],sigma)
            arrays[f'{budget}_{part}_mu']=mu;arrays[f'{budget}_{part}_sigma']=sigma
        reports.append(report)
    relative=f'projection/{name}_seed{seed}'
    core.save_json(relative+'.json',reports)
    np.savez_compressed(core.ROOT/(relative+'.npz'),**arrays)
    return relative
