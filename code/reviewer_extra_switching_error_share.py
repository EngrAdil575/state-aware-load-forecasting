import json, numpy as np, pandas as pd, xgboost as xgb
D='data'; meta=json.load(open(f'{D}/meta.json')); G=meta['GROUPS']; SC=meta['scale']
BASE=G['lags']+G['calendar_asset']; out={}
for H in [1,6,24]:
    d=pd.read_parquet(f'{D}/feat_h{H}.parquet'); tr,va,te=[(d.split==s).values for s in ['train','val','test']]
    y=d.load_s.values; c=d.AssetId.map(SC).values
    m=xgb.XGBRegressor(n_estimators=2000,learning_rate=0.03,max_depth=6,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,early_stopping_rounds=100,random_state=42,n_jobs=2,tree_method='hist')
    m.fit(d.loc[tr,BASE],y[tr],eval_set=[(d.loc[va,BASE],y[va])],verbose=False)
    for split,msk in [('val',va),('test',te)]:
        dd=d[msk].copy(); p=np.maximum(m.predict(d.loc[msk,BASE]),0)
        e2=((dd.load_s.values-p)*c[msk])**2
        prev=d.groupby('AssetId').load_s.shift(1)[msk].values  # previous hour on continuous grid? use base
        out[(H,split)]=(e2,dd)
    # switching definition: |z_t - z_{t-1}|>0.2 on continuous hourly grid
base=pd.read_parquet(f'{D}/base.parquet')[['AssetId','ts','load_s']].sort_values(['AssetId','ts'])
base['prev']=base.groupby('AssetId').load_s.shift(1)
res={}
for (H,split),(e2,dd) in out.items():
    mm=dd[['AssetId','ts']].merge(base[['AssetId','ts','load_s','prev']],on=['AssetId','ts'],how='left')
    sw=(np.abs(mm.load_s-mm.prev)>0.2).values; ok=mm.prev.notna().values
    res[f'{H}_{split}']=dict(share_rows=100*sw[ok].mean(), share_sqerr=100*e2[ok&sw].sum()/e2[ok].sum(), n=int(ok.sum()))
    print(H,split,{k:round(v,1) for k,v in res[f'{H}_{split}'].items()})
json.dump(res,open('results_updated/switching_error_share.json','w'),indent=1)
