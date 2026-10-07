"""Separate old-pool embedding effects from changes to active-sampling decisions."""
import json
from pathlib import Path
import numpy as np
import oct_core as core
import oct_surrogates as models

with np.load(core.ROOT/'old_cross_inputs.npz') as data:
    y,length,domains,groups=[data[k].copy() for k in ['y','length','domains','groups']]
    features={v:data[v].astype(float) for v in ['mini_mean','bge_cls']}
    split={p:data[p].copy() for p in ['validation','test']}
    saved={k:data[k].copy() for k in data.files if k.startswith(('selected_','expected_'))}
records=[];files=[]
for seed in range(100,105):
    for sampling in features:
        selected=saved[f'selected_{sampling}_{seed}'].astype(int)
        for encoder,X in features.items():
            fitted=models.fit_predictor('gp_matern32',X[selected],y[selected],domains[selected],length[selected],seed,groups[selected])
            mu,sigma=models.predict(fitted,X,domains,length)
            metrics={}
            for part in ['validation','test']:
                query=np.asarray(split[part])
                report=models.quality(y[query],mu[query],domains[query],y[selected],domains[selected],sigma[query])
                metrics[part]=dict(mae=report['__all__']['mae'],r2=report['__all__']['r2'],
                    theta_error=float(np.mean([r['theta']['0.05']['absolute_error'] for d,r in report.items() if d!='__all__'])))
            if sampling==encoder:
                for part in ['validation','test']:
                    query=np.asarray(split[part])
                    assert np.max(abs(mu[query]-saved[f'expected_{encoder}_{seed}_{part}_mu']))<1e-12
                    assert np.max(abs(sigma[query]-saved[f'expected_{encoder}_{seed}_{part}_sigma']))<1e-12
            records.append(dict(seed=seed,sampling=sampling,encoder=encoder,metrics=metrics))
            name=f'fresh_product/old_cross/{sampling}_{encoder}_{seed}.npz'
            core.ROOT.joinpath(name).parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(core.ROOT/name,mu=mu,sigma=sigma,selected=selected,y=y,domains=domains,
                validation=np.asarray(split['validation']),test=np.asarray(split['test']))
            files.append(name)
aggregates=[]
for sampling in features:
    for encoder in features:
        for part in ['validation','test']:
            chosen=[r['metrics'][part] for r in records if r['sampling']==sampling and r['encoder']==encoder]
            aggregates.append(dict(sampling=sampling,encoder=encoder,part=part,
                **{k:float(np.mean([r[k] for r in chosen])) for k in ['mae','r2','theta_error']}))
name='fresh_product/old_cross/summary.json'
core.save_json(name,dict(job_id=GP_JOB_ID,scope='Diagnostic of the already-inspected old product pool; not independent confirmation.',
    cases=20,records=records,aggregates=aggregates,same_encoder_sampling_regression_tolerance=1e-12,new_qwen_measurements=0))
files.append(name);print(json.dumps(aggregates),flush=True);core.export(files,GP_JOB_ID)
