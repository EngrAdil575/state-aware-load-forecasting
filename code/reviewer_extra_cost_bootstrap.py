import sys, json, time, os, numpy as np, pandas as pd, xgboost as xgb
SEED=42; D='data'; meta=json.load(open(f'{D}/meta.json')); G=meta['GROUPS']; TH=meta['THETA']; SCALE=meta['scale']
BASE=G['lags']+G['calendar_asset']; F=BASE+G['q15']+['on_lag1','on_yday','hours_in_state']+G['prevday']+G['recent']
out={}
for H in [1,6,24]:
    d=pd.read_parquet(f'{D}/feat_h{H}.parquet'); tr,va,te=[(d.split==s).values for s in ['train','val','test']]
    y=d.load_s.values; c=d.AssetId.map(SCALE).values
    age=(d.ts.max()-d.ts).dt.days.values; w=np.exp(-(age[tr]-age[tr].min())/90.0)
    R={}
    def mk(**k): return xgb.XGBRegressor(n_estimators=2000,learning_rate=0.03,max_depth=6,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,early_stopping_rounds=100,random_state=SEED,n_jobs=2,**k)
    t=time.time(); xb=mk(tree_method='hist'); xb.fit(d.loc[tr,BASE],y[tr],eval_set=[(d.loc[va,BASE],y[va])],verbose=False); R['fit_xgb']=time.time()-t
    Xtr,Xva,Xte=d.loc[tr,F],d.loc[va,F],d.loc[te,F]; on_tr,on_va=(y[tr]>TH).astype(int),(y[va]>TH).astype(int)
    t=time.time()
    clf=xgb.XGBClassifier(n_estimators=1500,learning_rate=0.03,max_depth=5,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,early_stopping_rounds=100,eval_metric='logloss',random_state=SEED,n_jobs=2)
    clf.fit(Xtr,on_tr,sample_weight=w,eval_set=[(Xva,on_va)],verbose=False)
    def reg(mt,mv):
        r=mk(); r.fit(Xtr[mt],y[tr][mt],sample_weight=w[mt],eval_set=[(Xva[mv],y[va][mv])],verbose=False); return r
    ron,roff=reg(on_tr==1,on_va==1),reg(on_tr==0,on_va==0); R['fit_mix']=time.time()-t
    # batch inference
    t=time.time(); px=xb.predict(d.loc[te,BASE]); R['inf_xgb_batch_us']=(time.time()-t)/te.sum()*1e6
    t=time.time(); p=clf.predict_proba(Xte)[:,1]; pm=p*ron.predict(Xte)+(1-p)*roff.predict(Xte); R['inf_mix_batch_us']=(time.time()-t)/te.sum()*1e6
    # single-row latency (13 assets at one origin)
    one=Xte.iloc[:13]; oneb=d.loc[te,BASE].iloc[:13]
    t=time.time()
    for _ in range(100): xb.predict(oneb)
    R['lat_xgb_13_ms']=(time.time()-t)/100*1e3
    t=time.time()
    for _ in range(100): pp=clf.predict_proba(one)[:,1]; pp*ron.predict(one)+(1-pp)*roff.predict(one)
    R['lat_mix_13_ms']=(time.time()-t)/100*1e3
    # sizes and trees
    def size(m,nm): f=f'preds/_{nm}_h{H}.ubj'; m.get_booster().save_model(f); return os.path.getsize(f)
    R['size_xgb_kb']=size(xb,'xgb')/1024; R['size_mix_kb']=(size(clf,'clf')+size(ron,'ron')+size(roff,'roff'))/1024
    R['trees_xgb']=xb.best_iteration+1; R['trees_mix']=[clf.best_iteration+1,ron.best_iteration+1,roff.best_iteration+1]
    yk=y[te]*c[te]; ex=yk-np.maximum(px,0)*c[te]; em=yk-np.maximum(pm,0)*c[te]
    R['rmse_xgb']=float(np.sqrt(np.mean(ex**2))); R['rmse_mix']=float(np.sqrt(np.mean(em**2)))
    # weekly block bootstrap of the RMSE reduction
    ts=d.ts.values[te]; wk=((ts-ts.min())/np.timedelta64(7,'D')).astype(int)
    W=pd.DataFrame({'wk':wk,'ex':ex**2,'em':em**2}).groupby('wk').agg(sx=('ex','sum'),sm=('em','sum'),n=('ex','size'))
    rng=np.random.default_rng(0); nb=len(W); red=[]
    for b in range(5000):
        s=W.iloc[rng.integers(0,nb,nb)]
        red.append(100*(1-np.sqrt(s.sm.sum()/s.n.sum())/np.sqrt(s.sx.sum()/s.n.sum())))
    R['red']=100*(1-R['rmse_mix']/R['rmse_xgb']); R['ci']=list(np.percentile(red,[2.5,97.5])); R['weeks']=nb; R['n_test']=int(te.sum())
    out[H]=R; print(H,json.dumps(R,default=float),flush=True)
json.dump(out,open('results_updated/refine_seed42.json','w'),indent=1,default=float)
