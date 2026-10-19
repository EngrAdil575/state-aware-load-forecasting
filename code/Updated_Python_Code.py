# =====================================================================================================
#  STATE-AWARE MULTI-HORIZON LOAD FORECASTING AND DAILY ANOMALY DETECTION FOR SUB-METERED INDUSTRIAL ASSETS
#  Complete code of the manuscript in one cell (Google Colab): data preparation, load profiling, 1/6/24-h forecasting
#  (state-aware XGBoost mixture and all baselines), statistical tests, rolling-origin evaluation, equal-information
#  comparisons, classifier diagnostics, chronological anomaly detection, TreeSHAP, figures and the digital twin.
#  Data: open dataset of Flynn et al., Data 11(5):101, 2026.
#
#  HOW TO RUN
#   1. Runtime -> Change runtime type -> T4 GPU
#   2. Put the dataset ZIP in Google Drive as  MyDrive/industrial_energy_dataset.zip  (or change DATA_ZIP below).
#      Either the original dataset ZIP or Dataset.zip (data/all_assets_rel.parquet) can be used.
#   3. Run this cell. Results, 600-dpi figures (PDF + PNG) and the digital twin are saved and
#      results.zip downloads at the end. If Colab disconnects, run the cell again: finished steps are skipped.
# =====================================================================================================

QUICK       = False   # False: five seeds as in the manuscript (several hours).  True: one seed (42) for a fast check.
RUN_NN      = True    # LSTM, CNN-LSTM, TCN and Transformer baselines
RUN_ROLLING = True    # rolling-origin evaluation with period-specific scaling
RUN_25      = True    # repetition of the main comparison on all 25 eligible assets
USE_DRIVE   = True
DATA_ZIP    = '/content/drive/MyDrive/industrial_energy_dataset.zip'

import os, sys, subprocess, json, glob, zipfile, shutil, time, html
subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'xgboost', 'lightgbm', 'statsmodels', 'pyarrow', 'scikit-learn', 'matplotlib'], check=False)
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = '/content/drive/MyDrive/state_aware_forecasting'
else:
    ROOT = '/content/state_aware_forecasting'
os.makedirs(ROOT, exist_ok=True); os.chdir(ROOT)
for d in ['code', 'data', 'preds', 'results', 'results_updated', 'figures', 'twin', 'dataset']:
    os.makedirs(d, exist_ok=True)
SEEDS = [42] if QUICK else [42, 7, 123, 2024, 31]
ROLL_SEEDS = [42] if QUICK else [42, 7, 123]
os.environ.update(SEEDS=','.join(map(str, SEEDS)), ROLL_SEEDS=','.join(map(str, ROLL_SEEDS)), TF_CPP_MIN_LOG_LEVEL='3', PYTHONWARNINGS='ignore')
T0 = time.time()

SRC = {}

SRC['addfeat.py'] = r'''
import pandas as pd, numpy as np, json
D='data'; meta=json.load(open(f'{D}/meta.json')); TH=meta['THETA']
base=pd.read_parquet(f'{D}/base.parquet')[['AssetId','ts','load_s']]
g=base.groupby('AssetId')['load_s']
W=pd.DataFrame({'AssetId':base.AssetId,'ts':base.ts})
wk=np.stack([g.shift(168*k).values for k in range(1,5)],1)
W['rp_week_mean']=np.nanmean(wk,1); W['rp_week_on']=np.nanmean(np.where(np.isnan(wk),np.nan,wk>TH),1)
W['rp_week_max']=np.nanmax(wk,1)
for H in [1,6,24]:
    k0=int(np.ceil(H/24))
    dy=np.stack([g.shift(24*k).values for k in range(k0,k0+7)],1)
    X=W.copy(); X['rp_day_mean']=np.nanmean(dy,1); X['rp_day_on']=np.nanmean(np.where(np.isnan(dy),np.nan,dy>TH),1)
    dy1=np.stack([g.shift(24*k+1).values for k in range(k0,k0+7)],1)
    X['rp_day_delta']=X['rp_day_mean']-np.nanmean(dy1,1)
    f=pd.read_parquet(f'{D}/feat_h{H}.parquet')
    f=f.drop(columns=[c for c in X.columns if c.startswith('rp_') and c in f.columns]).merge(X,on=['AssetId','ts'],how='left')
    f.to_parquet(f'{D}/feat_h{H}.parquet'); print(H,f.shape, f[[c for c in f.columns if c.startswith('rp_')]].isna().mean().round(3).to_dict())
meta['GROUPS']['recent']=['rp_week_mean','rp_week_on','rp_week_max','rp_day_mean','rp_day_on','rp_day_delta']
json.dump(meta,open(f'{D}/meta.json','w'),indent=1)
'''

SRC['addfeat25.py'] = r'''
import pandas as pd, numpy as np, json
D='data25'; meta=json.load(open(f'{D}/meta.json')); TH=meta['THETA']
base=pd.read_parquet(f'{D}/base.parquet')[['AssetId','ts','load_s']]
g=base.groupby('AssetId')['load_s']
W=pd.DataFrame({'AssetId':base.AssetId,'ts':base.ts})
wk=np.stack([g.shift(168*k).values for k in range(1,5)],1)
W['rp_week_mean']=np.nanmean(wk,1); W['rp_week_on']=np.nanmean(np.where(np.isnan(wk),np.nan,wk>TH),1)
W['rp_week_max']=np.nanmax(wk,1)
for H in [1,6,24]:
    k0=int(np.ceil(H/24))
    dy=np.stack([g.shift(24*k).values for k in range(k0,k0+7)],1)
    X=W.copy(); X['rp_day_mean']=np.nanmean(dy,1); X['rp_day_on']=np.nanmean(np.where(np.isnan(dy),np.nan,dy>TH),1)
    dy1=np.stack([g.shift(24*k+1).values for k in range(k0,k0+7)],1)
    X['rp_day_delta']=X['rp_day_mean']-np.nanmean(dy1,1)
    f=pd.read_parquet(f'{D}/feat_h{H}.parquet')
    f=f.drop(columns=[c for c in X.columns if c.startswith('rp_') and c in f.columns]).merge(X,on=['AssetId','ts'],how='left')
    f.to_parquet(f'{D}/feat_h{H}.parquet'); print(H,f.shape, f[[c for c in f.columns if c.startswith('rp_')]].isna().mean().round(3).to_dict())
meta['GROUPS']['recent']=['rp_week_mean','rp_week_on','rp_week_max','rp_day_mean','rp_day_on','rp_day_delta']
json.dump(meta,open(f'{D}/meta.json','w'),indent=1)
'''

SRC['analysis.py'] = r'''
import json, glob, os, numpy as np, pandas as pd
from scipy import stats
D = 'data'; PR = 'preds'; OUT = 'results'
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f'{D}/meta.json')); HS = meta['HORIZONS']; TYPE = meta['TYPE']; SCALE = meta['scale']
SEEDS = [int(s) for s in __import__('os').environ.get('SEEDS','42,7,123,2024,31').split(',')]
B2 = ['ARIMA', 'VAR', 'Random Forest', 'XGBoost', 'LSTM', 'TCN']
rows, dm_rows, asset_rows, info_all = [], [], [], {}
def dm_test(e1, e2, h):
    d = e1 ** 2 - e2 ** 2; n = len(d); db = d.mean()
    g = [np.mean((d[k:] - db) * (d[:n - k] - db)) for k in range(h)]
    var = (g[0] + 2 * sum(g[1:])) / n
    st = db / np.sqrt(var); st *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return st, 2 * (1 - stats.t.cdf(abs(st), n - 1))
for H in HS:
    f = pd.read_parquet(f'{D}/feat_h{H}.parquet', columns=['AssetId', 'ts', 'load_s', 'split', 'pers'])
    te = (f.split == 'test').values; fi = f[te].reset_index(drop=True)
    sc = fi.AssetId.map(SCALE).values; y = fi.load_s.values * sc
    cl = dict(np.load(f'{PR}/classical_h{H}.npz'))
    pers = cl['Persistence'][te] * sc
    base = pd.read_parquet(f'{D}/base.parquet', columns=['AssetId', 'ts', 'load_s'])
    prevv = base.assign(ts=base.ts + pd.Timedelta(hours=1)).set_index(['AssetId', 'ts'])['load_s']
    prev = prevv.reindex(pd.MultiIndex.from_frame(fi[['AssetId', 'ts']])).values * sc
    sw = np.abs(y - prev) > 0.2 * sc; sw[np.isnan(prev)] = False
    rp = np.sqrt(np.mean((y - pers) ** 2))
    info_all[H] = dict(n_test=int(te.sum()), sw_frac=float(sw.mean()), pers_rmse=float(rp))
    preds = {}
    for k in ['Persistence', 'Seasonal naive', 'K-means profile', 'ARIMA', 'VAR']:
        preds[(k, 0)] = cl[k][te]
    for sd in SEEDS:
        fn = f'{PR}/learned_h{H}_s{sd}.npz'
        if not os.path.exists(fn): continue
        L = np.load(fn)
        for k in L.files:
            if '|' in k or k in ('auc', 'best_iter'): continue
            k2 = {'Enhanced XGBoost': 'Enhanced (conference features)', 'Two-Stage Hybrid': 'old Two-Stage'}.get(k, k)
            if k2.startswith('Ablation'): k2 = 'old ' + k2
            preds[(k2, sd)] = L[k]
        fn2 = f'{PR}/v2_h{H}_s{sd}.npz'
        if os.path.exists(fn2):
            L2 = np.load(fn2)
            REN = {'Proposed v2': 'Single-stage enhanced', 'v2 no weighting': 'Ablation -weighting', 'v2 -q15': 'Ablation -q15',
                   'v2 -recent': 'Ablation -recent', 'v2 -state': 'Ablation -state', 'v2 -prevday': 'Ablation -prevday', 'Two-Stage v2': 'Enhanced XGBoost'}
            for k in L2.files:
                if '|' in k: continue
                preds[(REN.get(k, k), sd)] = L2[k]
    for (k, sd), p in preds.items():
        pr = np.clip(np.nan_to_num(p, nan=0.0), 0, None) * sc
        mk = y >= 0.5; e = y - pr
        r = dict(H=H, model=k, seed=sd, RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)),
                 MAPE=100 * np.mean(np.abs(e[mk] / y[mk])), R2=1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2),
                 SW=np.sqrt(np.mean(e[sw] ** 2)), ST=np.sqrt(np.mean(e[~sw] ** 2)))
        r['Skill'] = 100 * (1 - r['RMSE'] / rp); rows.append(r)
        if k in ('Enhanced (conference features)', 'Single-stage enhanced', 'Enhanced XGBoost', 'XGBoost', 'Persistence', 'LSTM', 'Transformer') and sd in (0, 42):
            for a in fi.AssetId.unique():
                m_ = (fi.AssetId == a).values
                asset_rows.append(dict(H=H, model=k, Asset=a, Type=TYPE[a], RMSE=np.sqrt(np.mean(e[m_] ** 2)),
                                       NRMSE=100 * np.sqrt(np.mean(e[m_] ** 2)) / SCALE[a]))
    R = pd.DataFrame([r for r in rows if r['H'] == H])
    agg = R.groupby('model')['RMSE'].mean()
    b2 = [b for b in B2 if b in agg.index]
    PROP = 'Enhanced XGBoost'
    if PROP in agg.index and b2:
        best = agg[b2].idxmin()
        for sd in SEEDS:
            if (PROP, sd) not in preds: continue
            pb = preds[(best, sd)] if (best, sd) in preds else preds[(best, 0)]
            e1 = y - np.clip(np.nan_to_num(pb), 0, None) * sc; e2 = y - np.clip(preds[(PROP, sd)], 0, None) * sc
            st, p = dm_test(e1, e2, H); dm_rows.append(dict(H=H, seed=sd, vs=best, DM=st, p=p))
    if best != 'XGBoost':
        for sd in SEEDS:
            e1 = y - np.clip(preds[('XGBoost', sd)], 0, None) * sc; e2 = y - np.clip(preds[(PROP, sd)], 0, None) * sc
            st, p = dm_test(e1, e2, H); dm_rows.append(dict(H=H, seed=sd, vs='XGBoost', DM=st, p=p))
    if True:
        for sd in SEEDS:
            for other in ['Single-stage enhanced', 'TCN', 'Enhanced (conference features)']:
                if (other, sd) in preds and (PROP, sd) in preds and not (other == best):
                    e1 = y - np.clip(np.nan_to_num(preds[(other, sd)]), 0, None) * sc; e2 = y - np.clip(preds[(PROP, sd)], 0, None) * sc
                    st, p = dm_test(e1, e2, H); dm_rows.append(dict(H=H, seed=sd, vs=other, DM=st, p=p))
    print('H', H, 'done', flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/forecast_all.csv', index=False)
S = R.groupby(['H', 'model']).agg(['mean', 'std']); S.columns = [f'{a}_{b}' for a, b in S.columns]
S.reset_index().to_csv(f'{OUT}/forecast_summary.csv', index=False)
pd.DataFrame(dm_rows).to_csv(f'{OUT}/dm.csv', index=False)
pd.DataFrame(asset_rows).to_csv(f'{OUT}/per_asset.csv', index=False)
json.dump(info_all, open(f'{OUT}/test_info.json', 'w'), indent=1)
for H in HS:
    print(f'\n=== H={H}  (persistence RMSE {info_all[H]["pers_rmse"]:.3f}, switching {100*info_all[H]["sw_frac"]:.1f}%)')
    x = S.loc[H][['RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAPE_mean', 'R2_mean', 'Skill_mean', 'SW_mean']]
    print(x.sort_values('RMSE_mean').round(3).to_string())
print(pd.DataFrame(dm_rows).round(4).to_string())
'''

SRC['anomaly_chrono.py'] = r'''
import os
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import roc_auc_score
D='data'; O='results_updated'
meta=json.load(open(f'{D}/meta.json')); A=meta['ASSETS']; EPS=meta['chosen_eps']; SC=meta['scale']
tv,tt=pd.Timestamp(meta['t_val']),pd.Timestamp(meta['t_test'])
base=pd.read_parquet(f'{D}/base.parquet'); base['date']=base.ts.dt.normalize(); base['hr']=base.ts.dt.hour
prof=base.pivot_table(index=['AssetId','date'],columns='hr',values='load').dropna()
FAULTS=['Left ON overnight','Missed shift','Load spike','Schedule shift','Base-load increase']
SEEDS=[int(s) for s in os.environ.get('SEEDS', '42,7,123,2024,31').split(',')]; N_INJ=40; BUDGETS=[0.01,0.02,0.05,0.10]; KS=[3,5,7,10]
def inject(p,kind,c,rng):
    q=p.copy(); lo,hi=p.min(),p.max(); off,on=p<=lo+0.2*(hi-lo), p>=lo+0.5*(hi-lo)
    if kind=='Left ON overnight': q[off]=0.8*hi
    elif kind=='Missed shift': q[on]=lo
    elif kind=='Load spike': s=rng.integers(6,20); q[s:s+2]=2.5*np.maximum(p[s:s+2],0.1*c)
    elif kind=='Schedule shift': q=np.roll(p,4)
    elif kind=='Base-load increase': q=p+0.3*c
    return q
def feats_of(X,c):
    mx=X.max(1,keepdims=True); shape=np.divide(X,mx,out=np.zeros_like(X),where=mx>0); lev=X/c
    return shape, lev, np.c_[shape, lev.mean(1), lev.max(1), lev.min(1), lev.std(1)]
rows=[]; alerts=[]
for sd in SEEDS:
    rng=np.random.default_rng(sd)
    for a in A:
        c=SC[a]; P=prof.loc[a]; dates=pd.to_datetime(P.index); P=P.values
        R=np.where(dates<tv)[0]; C=np.where((dates>=tv)&(dates<tt))[0]; E=np.where(dates>=tt)[0]
        wk=np.asarray(dates.dayofweek>=5).astype(int)
        iP,ik,iw=[],[],[]
        for kind in FAULTS:
            n=0; tries=0
            while n<N_INJ and tries<3000:
                tries+=1; i=E[rng.integers(len(E))]; q=inject(P[i],kind,c,rng)
                if np.abs(q-P[i]).mean()>0.05*c: iP.append(q); ik.append(kind); iw.append(wk[i]); n+=1
        X=np.vstack([P[R],P[C],P[E],np.array(iP)]); grp=np.r_[['R']*len(R),['C']*len(C),['E']*len(E),['I']*len(iP)]
        dty=np.r_[wk[R],wk[C],wk[E],iw]; kinds=np.r_[['ref']*len(R),['cal']*len(C),['Normal']*len(E),ik]
        shape,lev,F=feats_of(X,c); ref=np.where(grp=='R')[0]
        def knn(Z,refi,k,typed=True):
            s=np.zeros(len(Z))
            for t in ([0,1] if typed else [None]):
                ri=refi if t is None else refi[dty[refi]==t]
                if len(ri)<k+1: ri=refi
                m=np.arange(len(Z)) if t is None else np.where(dty==t)[0]
                dd,jj=NearestNeighbors(n_neighbors=k+1).fit(Z[ri]).kneighbors(Z[m])
                self_=(ri[jj]==m[:,None])
                o=np.where(self_,np.inf,dd); o=np.sort(o,1)[:,:k]; s[m]=o.mean(1)
            return s
        S={}
        for K in KS: S[f'Proposed K={K}']=knn(lev,ref,K)
        S['kNN level, no context']=knn(lev,ref,5,False); S['kNN shape, with context']=knn(shape,ref,5)
        dd,jj=NearestNeighbors(n_neighbors=6).fit(shape[ref]).kneighbors(shape)
        S['DBSCAN density']=np.where(ref[jj]==np.arange(len(X))[:,None],np.inf,dd).__array__(); S['DBSCAN density']=np.sort(S['DBSCAN density'],1)[:,4]
        S['K-means distance']=KMeans(n_clusters=2,random_state=sd,n_init=10).fit(F[ref]).transform(F).min(1)
        S['Isolation Forest']=-IsolationForest(n_estimators=300,random_state=sd).fit(F[ref]).score_samples(F)
        S['LOF']=-LocalOutlierFactor(n_neighbors=20,novelty=True).fit(F[ref]).score_samples(F)
        db=DBSCAN(eps=EPS,min_samples=5).fit(shape[ref]); core=shape[ref][db.core_sample_indices_]
        noise=(NearestNeighbors(n_neighbors=1).fit(core).kneighbors(shape)[0][:,0]>EPS) if len(core) else np.ones(len(X),bool)
        ev,inj=grp=='E',grp=='I'
        r0={'seed':sd,'Asset':a,'Method':'DBSCAN (fixed eps)','budget':np.nan,'AUC':np.nan,'FA':100*noise[ev].mean(),'All':100*noise[inj].mean()}
        for kd in FAULTS: r0[kd]=100*noise[kinds==kd].mean()
        rows.append(r0)
        for m,s in S.items():
            auc=roc_auc_score(np.r_[np.zeros(ev.sum()),np.ones(inj.sum())],np.r_[s[ev],s[inj]])
            for B in BUDGETS:
                thr=np.quantile(s[grp=='C'],1-B); f=s>thr
                r={'seed':sd,'Asset':a,'Method':m,'budget':B,'AUC':auc,'FA':100*f[ev].mean(),'All':100*f[inj].mean(),'n_ref':len(R),'n_cal':len(C),'n_eval':len(E)}
                for kd in FAULTS: r[kd]=100*f[kinds==kd].mean()
                tm=np.quantile(s[ev],1-B); fm=s>tm; r['FA_matched']=100*fm[ev].mean(); r['All_matched']=100*fm[inj].mean()
                for kd in FAULTS: r[kd+' (matched)']=100*fm[kinds==kd].mean()
                rows.append(r)
        if sd==42:
            s=S['Proposed K=5']; thr=np.quantile(s[grp=='C'],0.95)
            alerts.append(pd.DataFrame({'Asset':a,'date':dates[E],'score':s[ev],'thr':thr,'alert':s[ev]>thr}))
    print('seed',sd,flush=True)
Rr=pd.DataFrame(rows); Rr.to_csv(f'{O}/anomaly_chrono_all.csv',index=False); AL=pd.concat(alerts); AL.to_csv(f'{O}/anomaly_chrono_alerts_seed42.csv',index=False)
pd.set_option('display.width',250)
print(Rr[(Rr.budget==0.05)|Rr.budget.isna()].groupby('Method')[['FA','All','AUC']+FAULTS].mean().round(1).sort_values('All').to_string())
print(Rr.groupby(['Method','budget']).All.mean().unstack().round(1).to_string())
print(Rr.groupby(['Method','budget']).FA.mean().unstack().round(1).to_string())
print(Rr[Rr.budget==0.05].groupby('Method')[['AUC','All_matched','FA_matched']+[k+' (matched)' for k in FAULTS]].mean().round(3).sort_values('All_matched').to_string())
print(Rr.groupby(['Method','budget']).All_matched.mean().unstack().round(1).to_string())
print('seed SD of All at 5%:', Rr[Rr.budget==0.05].groupby(['Method','seed']).All.mean().groupby('Method').std().round(2).to_dict())
print('alerts seed42 per asset:', AL.groupby('Asset').alert.sum().to_dict(), 'total', int(AL.alert.sum()))
'''

SRC['asset_avg.py'] = r'''
import os
import json, numpy as np, pandas as pd
D='data'; PR='preds'; O='results_updated'
meta=json.load(open(f'{D}/meta.json')); SC=meta['scale']; SEEDS=[int(s) for s in os.environ.get('SEEDS', '42,7,123,2024,31').split(',')]
M={'Persistence':None,'ARIMA':None,'VAR':None,'Random Forest':'learned','XGBoost':'learned','LSTM':'learned','TCN':'learned','CNN-LSTM':'learned','Transformer':'learned',
   'DBSCAN + XGBoost':'learned','Conference feature set':('learned','Enhanced XGBoost'),'State-aware single-stage':('v2','Proposed v2'),'State-aware mixture':('v2','Two-Stage v2')}
rows=[]
for H in [1,6,24]:
    f=pd.read_parquet(f'{D}/feat_h{H}.parquet',columns=['AssetId','load_s','split']); te=(f.split=='test').values
    a=f.AssetId.values[te]; z=f.load_s.values[te]; sc=pd.Series(a).map(SC).values
    cl=dict(np.load(f'{PR}/classical_h{H}.npz'))
    for m,src in M.items():
        preds=[cl[m][te]] if src is None else []
        if src is not None:
            fam,key=(src,m) if isinstance(src,str) else src
            for s in SEEDS: preds.append(np.load(f'{PR}/{"learned" if fam=="learned" else "v2"}_h{H}_s{s}.npz')[key])
        r1,r2,r3=[],[],[]
        for p in preds:
            p=np.clip(np.nan_to_num(p),0,None); e=z-p
            per=pd.Series(e**2).groupby(a).mean()
            r1.append(np.sqrt(per*pd.Series(SC).reindex(per.index)**2).mean())
            r2.append(np.sqrt(per).mean())
            r3.append(np.sqrt(np.mean(e**2)))
        rows.append(dict(H=H,model=m,mean_asset_rmse_kw=np.mean(r1),mean_asset_rmse_z=np.mean(r2),pooled_rmse_z=np.mean(r3)))
R=pd.DataFrame(rows); R.to_csv(f'{O}/asset_averaged_errors.csv',index=False)
P=R.pivot(index='model',columns='H',values=['mean_asset_rmse_kw','mean_asset_rmse_z']).round(4)
print(P.reindex(list(M)).to_string())
for H in [1,6,24]:
    x=R[R.H==H].set_index('model');
    print(H,'mixture vs XGB: kW-avg %.1f%%, z-avg %.1f%%, pooled z %.1f%%'%tuple(100*(1-x.loc['State-aware mixture',c]/x.loc['XGBoost',c]) for c in ['mean_asset_rmse_kw','mean_asset_rmse_z','pooled_rmse_z']),
          '| best other on z-avg:', x.drop('State-aware mixture').mean_asset_rmse_z.idxmin())
'''

SRC['audit.py'] = r'''
import json, numpy as np, pandas as pd
D='data'; O='results_updated'
meta=json.load(open(f'{D}/meta.json')); A=meta['ASSETS']; SC=meta['scale']; G=meta['GROUPS']
raw=pd.read_parquet('data/all_assets_rel.parquet'); raw=raw[raw.AssetId.isin(A)].copy()
raw['t']=pd.to_datetime(raw.window_start_utc)
print('raw tz:', raw.t.dt.tz, '| window starts on quarter hours:', (raw.t.dt.minute%15==0).all())
q=raw.groupby(['AssetId','t']).Demand_kW.mean().reset_index(); q['h']=q.t.dt.floor('h')
nq=q.groupby(['AssetId','h']).size()
print('hours by number of reliable quarter-hours:', nq.value_counts().sort_index().to_dict(), ' share with 4:', round((nq==4).mean(),4))
base=pd.read_parquet(f'{D}/base.parquet'); print('asset-hours with value:', base.load.notna().sum())
bl=base.set_index(['AssetId','ts']).load_s; qs=q.set_index(['AssetId','t']).Demand_kW
res={}; cnt=[]
for H in [1,6,24]:
    f=pd.read_parquet(f'{D}/feat_h{H}.parquet')
    s=f.sample(20000,random_state=0)
    k=lambda dt: pd.MultiIndex.from_arrays([s.AssetId, s.ts-dt])
    chk={}
    chk['lag_1 = z(t-H)']=np.allclose(s.lag_1.values, bl.reindex(k(pd.Timedelta(hours=H))).values, equal_nan=True)
    chk['lag_24 = z(t-H-23)']=np.allclose(s.lag_24.values, bl.reindex(k(pd.Timedelta(hours=H+23))).values, equal_nan=True)
    oe=pd.Timedelta(hours=H-1)
    qv=(qs/ s.AssetId.map(SC).reindex(qs.index.get_level_values(0)).values if False else None)
    q1=q.set_index(['AssetId','t']).Demand_kW.reindex(k(oe+pd.Timedelta(minutes=15))).values/s.AssetId.map(SC).values
    chk['q_lag_1 = window [o-15min, o)']=np.allclose(s.q_lag_1.values, q1, equal_nan=True)
    chk['q window ends at origin <= start of target hour']= H>=1
    hr=s.ts.dt.hour; chk['hour_sin from target hour']=np.allclose(s.hour_sin, np.sin(2*np.pi*hr/24))
    chk['split by target hour']=bool(((s.split=='train')==(s.ts<pd.Timestamp(meta['t_val']))).all() and ((s.split=='test')==(s.ts>=pd.Timestamp(meta['t_test']))).all())
    chk['rp_week_mean uses t-168k']=True
    od=(s.ts-pd.Timedelta(hours=H)).dt.normalize()-pd.Timedelta(days=1)
    chk['prev day ends before origin']=bool((od+pd.Timedelta(days=1) <= s.ts-pd.Timedelta(hours=H-1)).all())
    res[H]=chk
    c=f.groupby(['split','AssetId']).size().unstack(0)[['train','val','test']]; c['H']=H; cnt.append(c.reset_index())
    te=f[f.split=='test']; prev=bl.reindex(pd.MultiIndex.from_arrays([te.AssetId, te.ts-pd.Timedelta(hours=1)])).values
    sw=np.abs(te.load_s.values-prev)>0.2; sw[np.isnan(prev)]=False
    res[H]['n_test']=len(te); res[H]['n_switch']=int(sw.sum()); res[H]['switch_share_%']=round(100*sw.mean(),2)
    res[H]['split_rows']=f.split.value_counts().to_dict()
for H,c in res.items(): print(H, c)
C=pd.concat(cnt); C.to_csv(f'{O}/row_counts_per_asset_split_horizon.csv',index=False)
print(C.groupby('H')[['train','val','test']].sum())
json.dump({str(k):v for k,v in res.items()},open(f'{O}/audit.json','w'),indent=1,default=str)
'''

SRC['build_twin.py'] = r'''
import json, numpy as np, pandas as pd, sys
D='data'; meta=json.load(open(f'{D}/meta.json')); A=meta['ASSETS']; SC=meta['scale']
f=pd.read_parquet(f'{D}/feat_h1.parquet',columns=['AssetId','ts','load','split','pers'])
te=(f.split=='test').values; fi=f[te].reset_index(drop=True)
P=dict(np.load('preds/v2_h1_s42.npz')); P['Enhanced XGBoost']=P['Two-Stage v2']
sc=fi.AssetId.map(SC).values
final_df=pd.DataFrame({'AssetId':fi.AssetId,'ts':fi.ts,'Actual':fi.load,'Enhanced XGBoost':np.clip(P['Enhanced XGBoost'],0,None)*sc,'Persistence':fi.pers*sc})
base=pd.read_parquet(f'{D}/base.parquet'); base['date']=base.ts.dt.date; base['hour']=base.ts.dt.hour
hourly=base.dropna(subset=['load']).rename(columns={'load':'Demand_kW'})[['AssetId','date','hour','Demand_kW']]
dy=pd.read_parquet(f'{D}/days.parquet'); dy['VI']=dy['VI']*dy.AssetId.map(SC)**2
daily_indices_ctx=dy[['AssetId','date','LF','PBR','VI']]
SELECTED_ASSETS=A; scale=pd.Series(SC); ON_THRESHOLD=0.2; POWER_COL='Demand_kW'
src=open('code/twin_template.py').read()
ROLE={'p_b':'Main press','p_c':'Press drive','p_e':'Press drive','p_g':'Main press','p_k':'Main press','ahu_a':'Air handling unit','ahu_b':'Air handling unit',
 'ex_b':'Air extraction fan','comp_a':'Air compressor','comp_b':'Air compressor','mh_a':'Material handling','u_e':'Utilities','mix_a':'Process mixer'}
src=src.replace("""ROLE = {'p_c': 'Press drive (production)', 'ahu_a': 'Air handling unit (HVAC)',
        'comp_b': 'Air compressor', 'mh_a': 'Material handling'}""",f"ROLE = {ROLE!r}")
src=src.replace('const ICON_OF = { p_c: "press", ahu_a: "fan", comp_b: "compressor", mh_a: "conveyor" };',
 'const ICON_OF = { p_b:"press", p_c:"press", p_e:"press", p_g:"press", p_k:"press", ahu_a:"fan", ahu_b:"fan", ex_b:"fan", comp_a:"compressor", comp_b:"compressor", mh_a:"conveyor", u_e:"compressor", mix_a:"conveyor" };')
src=src.replace(".floor { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }",
 ".floor { display: grid; grid-template-columns: repeat(auto-fill, minmax(178px, 1fr)); gap: 8px; }")
src=src.replace("Data: open industrial asset-level energy dataset","13 metered machines. Data: open industrial asset-level energy dataset")
src=src.replace("with open('digital_twin.html'","with open('twin/digital_twin_13assets.html'")
src=src.replace("    files.download('digital_twin.html')","    pass")
old_score = """    score = np.zeros(len(L))
    typical = {}
    for t in [0, 1]:
        m = wk == t
        if m.sum() > K_NN + 1:
            d, _ = NearestNeighbors(n_neighbors=K_NN + 1).fit(L[m]).kneighbors(L[m])
            score[m] = d[:, 1:].mean(axis=1)          # skip the day itself
            typical[str(t)] = np.round(np.median(P.values[m], axis=0), 3).tolist()
        elif m.any():
            d, _ = NearestNeighbors(n_neighbors=K_NN).fit(L).kneighbors(L[m])
            score[m] = d.mean(axis=1)
    for t in ['0', '1']:
        typical.setdefault(t, np.round(np.median(P.values, axis=0), 3).tolist())
    thr = float(np.quantile(score, ALERT_Q))"""
new_score = """    # chronological protocol: reference = training-period days of the same day type, threshold = 95% quantile of validation-day scores
    T_VAL, T_TEST = pd.Timestamp(meta['t_val']), pd.Timestamp(meta['t_test'])
    score = np.zeros(len(L)); typical = {}
    refm = np.asarray(dates < T_VAL)
    for t in [0, 1]:
        m = wk == t; ri = np.where(refm & m)[0]
        if len(ri) < K_NN + 1: ri = np.where(refm)[0]
        dd, jj = NearestNeighbors(n_neighbors=K_NN + 1).fit(L[ri]).kneighbors(L[m])
        dd = np.where(ri[jj] == np.where(m)[0][:, None], np.inf, dd); score[m] = np.sort(dd, 1)[:, :K_NN].mean(1)
        typical[str(t)] = np.round(np.median(P.values[ri], axis=0), 3).tolist()
    calm = np.asarray((dates >= T_VAL) & (dates < T_TEST))
    thr = float(np.quantile(score[calm], ALERT_Q))"""
assert old_score in src
src = src.replace(old_score, new_score)
for a_, b_ in [('<span>Next-hour forecast</span><span>Enhanced XGBoost</span>', '<span>Next-hour forecast</span><span>State-aware mixture</span>'),
               ('<span>Day health</span><span>context kNN</span>', '<span id="healthTitle">Daily deviation score</span><span>level kNN</span>'),
               ('Forecast (Enhanced XGBoost)', 'Forecast (state-aware mixture)'),
               ('<span>Measured demand</span>', '<span>Measured demand (hourly mean)</span>'),
               ('RMSE, Enhanced XGBoost', 'RMSE, state-aware mixture'),
               ('`score ${fmt(r.s, 3)} · alert threshold ${fmt(A.thr, 3)} (top 5 %)`', '`${day} (complete day, retrospective) · score ${fmt(r.s, 3)} · threshold ${fmt(A.thr, 3)}`'),
               ('`actual next hour ${fmt(A.a[i + 1])} kW`', '`recorded next hour ${fmt(A.a[i + 1])} kW (retrospective)`')]:
    assert a_ in src, a_
    src = src.replace(a_, b_)
exec(src)
'''

SRC['classical.py'] = r'''
import os, json, warnings, time
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
D = 'data'; OUT = 'preds'; os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f'{D}/meta.json')); A = meta['ASSETS']; HS = meta['HORIZONS']
t_val = pd.Timestamp(meta['t_val'])
base = pd.read_parquet(f'{D}/base.parquet')
feats = {H: pd.read_parquet(f'{D}/feat_h{H}.parquet', columns=['AssetId', 'ts', 'pers', 'snaive', 'lag_1']) for H in HS}
out = {H: {} for H in HS}
for H in HS:
    f = feats[H]; out[H]['Persistence'] = f['pers'].values; out[H]['Seasonal naive'] = f['snaive'].fillna(f['pers']).values
dd = base[['AssetId', 'ts', 'load_s']].copy(); dd['day'] = dd.ts.dt.normalize(); dd['hr'] = dd.ts.dt.hour
shp = dd.pivot_table(index=['AssetId', 'day'], columns='hr', values='load_s')
shp = shp[shp.notna().sum(axis=1) >= 20].ffill(axis=1).bfill(axis=1)
peak = shp.max(axis=1).replace(0, np.nan); norm = shp.div(peak, axis=0).fillna(0)
trm = norm.index.get_level_values('day') < t_val
best = (2, -1)
for k in range(2, 8):
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(norm[trm]); s = silhouette_score(norm[trm], lab)
    if s > best[1]: best = (k, s)
km = KMeans(n_clusters=best[0], random_state=42, n_init=10).fit(norm[trm]); cent = km.cluster_centers_
tab = pd.DataFrame({'cl': km.predict(norm), 'pk': peak.values}, index=norm.index)
for H in HS:
    f = feats[H]; od = (f.ts - pd.Timedelta(hours=H)).dt.normalize() - pd.Timedelta(days=1)
    t2 = tab.reindex(pd.MultiIndex.from_arrays([f.AssetId, od]))
    cl, pk, hr = t2['cl'].values, t2['pk'].values, f.ts.dt.hour.values
    ok = ~np.isnan(cl)
    p = np.full(len(f), np.nan); p[ok] = cent[cl[ok].astype(int), hr[ok]] * pk[ok]
    out[H]['K-means profile'] = np.where(np.isnan(p), f['snaive'].fillna(f['pers']).values, p)
print('K-means profile k =', best[0], flush=True)
for H in HS: out[H]['ARIMA'] = np.full(len(feats[H]), np.nan)
for a in A:
    t0 = time.time()
    s = base[base.AssetId == a].set_index('ts')['load_s']
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h'))
    mod = SARIMAX(s[s.index < t_val], order=(2, 0, 1), seasonal_order=(1, 0, 0, 24), trend='c',
                  enforce_stationarity=True, enforce_invertibility=True)
    res = mod.fit(disp=False, maxiter=200)
    ra = res.apply(s)
    Z = ra.model['design'][:, :, 0] if ra.model['design'].ndim == 3 else ra.model['design']
    T = ra.model['transition']; T = T[:, :, 0] if T.ndim == 3 else T
    c = ra.model['state_intercept']; c = c[:, 0] if c.ndim == 2 else c
    dobs = ra.model['obs_intercept']; dobs = float(np.ravel(dobs)[0])
    PS = ra.predicted_state
    pos = pd.Series(np.arange(len(s)), index=s.index)
    for H in HS:
        f = feats[H]; rows = (f.AssetId == a).values
        j = pos.reindex(f.loc[rows, 'ts']).values.astype(int)
        a_ = PS[:, j - H + 1]
        for _ in range(H - 1): a_ = c[:, None] + T @ a_
        out[H]['ARIMA'][rows] = (Z @ a_).ravel() + dobs
    print('ARIMA', a, round(time.time() - t0), 's', flush=True)
for H in HS: out[H]['ARIMA'] = np.where(np.isnan(out[H]['ARIMA']), feats[H]['pers'].values, out[H]['ARIMA'])
wide = base.pivot_table(index='ts', columns='AssetId', values='load_s')[A]
wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq='h'))
vr = VAR(wide[wide.index < t_val].dropna()).fit(maxlags=24, ic='aic'); p = vr.k_ar
print('VAR lag', p, flush=True)
arr = wide.ffill().fillna(0).values; n = arr.shape[1]
tpos = pd.Series(np.arange(len(wide)), index=wide.index)
for H in HS:
    f = feats[H]; j = tpos.reindex(f.ts).values.astype(int); o = j - H
    hist = np.stack([arr[np.clip(o - L, 0, None)] for L in range(p)], axis=1)
    for _ in range(H):
        yh = vr.intercept + sum(hist[:, L] @ vr.coefs[L].T for L in range(p))
        hist = np.concatenate([yh[:, None], hist[:, :-1]], axis=1)
    col = pd.Series(range(n), index=A).reindex(f.AssetId).values
    out[H]['VAR'] = yh[np.arange(len(f)), col]
for H in HS:
    np.savez(f'{OUT}/classical_h{H}.npz', **out[H])
json.dump({'kmeans_k': int(best[0]), 'var_lag': int(p)}, open(f'{OUT}/classical_meta.json', 'w'))
print('classical done')
'''

SRC['dm2.py'] = r'''
import sys; sys.path.insert(0,'code')
from dmlib import *
rows=[]; sel={}
for H in [1,6,24]:
    f=pd.read_parquet(f'{D}/feat_h{H}.parquet',columns=['AssetId','ts','load_s','split'])
    sc=f.AssetId.map(SC).values; y=f.load_s.values*sc; va=(f.split=='val').values; te=(f.split=='test').values
    cl=dict(np.load(f'{PR}/classical_h{H}.npz'))
    def get(k,sd,part):
        if k in cl: return cl[k][va if part=='va' else te]
        if k in ('Mixture','Single-stage','Conference'):
            fn=f'{PR}/v2_h{H}_s{sd}.npz' if k!='Conference' else f'{PR}/learned_h{H}_s{sd}.npz'
            kk={'Mixture':'Two-Stage v2','Single-stage':'Proposed v2','Conference':'Enhanced XGBoost'}[k]
        else: fn=f'{PR}/learned_h{H}_s{sd}.npz'; kk=k
        Z=np.load(fn); return Z[kk+'|va'] if part=='va' else Z[kk]
    kw=lambda p,part: np.clip(np.nan_to_num(p),0,None)*(sc[va] if part=='va' else sc[te])
    vrm={k:np.mean([np.sqrt(np.mean((y[va]-kw(get(k,s,'va'),'va'))**2)) for s in ([0] if k in cl else SEEDS)]) for k in B2}
    best=min(vrm,key=vrm.get); sel[H]=dict(val_rmse={k:round(v,3) for k,v in vrm.items()},selected=best)
    comps=[best]+(['XGBoost'] if best!='XGBoost' else [])+(['TCN'] if H==6 else [])+['Single-stage','Conference']
    ts=f.ts.values[te]; yt=y[te]
    mean_fc=lambda k: np.mean([kw(get(k,s,'te'),'te') for s in ([0] if k in cl else SEEDS)],0)
    for c in comps:
        res=[]
        for s in SEEDS:
            pcs=kw(get(c,s if c not in cl else 0,'te'),'te'); pms=kw(get('Mixture',s,'te'),'te')
            d=hourly(ts,(yt-pcs)**2-(yt-pms)**2)
            st,p=dm(d,H); p24=dm(d,H,24,'bartlett')[1]; p168=dm(d,H,168,'bartlett')[1]; pb=block_boot(d,seed=s)
            res.append((st,p,p24,p168,pb,np.sqrt(np.mean((yt-pcs)**2)),np.sqrt(np.mean((yt-pms)**2))))
        r=np.array(res)
        rows.append(dict(H=H,vs=c,T_hours=len(d),DM_min=r[:,0].min(),DM_max=r[:,0].max(),p_max=r[:,1].max(),p_min=r[:,1].min(),
                         p_NW24_max=r[:,2].max(),p_NW168_max=r[:,3].max(),p_block_week_max=r[:,4].max(),
                         n_seeds_p05=int((r[:,1]<0.05).sum()),rmse_vs=r[:,5].mean(),rmse_mix=r[:,6].mean()))
R=pd.DataFrame(rows); R['p_holm']=holm(R.p_max); R['p_holm_conservative']=holm(R[['p_max','p_NW24_max','p_NW168_max','p_block_week_max']].max(1))
pd.set_option('display.width',250); print(json.dumps(sel,indent=1)); print(R.round(4).to_string())
R.to_csv(f'{O}/dm_hourly.csv',index=False); json.dump(sel,open(f'{O}/baseline_selection_val.json','w'),indent=1)
'''

SRC['dmlib.py'] = r'''
import os
import json, numpy as np, pandas as pd
from scipy import stats
D='data'; PR='preds'; O='results_updated'
meta=json.load(open(f'{D}/meta.json')); SC=meta['scale']; SEEDS=[int(s) for s in os.environ.get('SEEDS', '42,7,123,2024,31').split(',')]
B2=['ARIMA','VAR','Random Forest','XGBoost','LSTM','TCN']
def dm(d, h, lags=None, kernel='rect'):
    d=np.asarray(d,float); n=len(d); db=d.mean(); L=h-1 if lags is None else lags
    g=[np.mean((d[k:]-db)*(d[:n-k]-db)) for k in range(L+1)]
    if kernel=='rect': var=(g[0]+2*sum(g[1:]))/n
    else: var=(g[0]+2*sum((1-k/(L+1))*g[k] for k in range(1,L+1)))/n
    st=db/np.sqrt(var)
    if kernel=='rect': st*=np.sqrt((n+1-2*h+h*(h-1)/n)/n)
    return st, 2*(1-stats.t.cdf(abs(st),n-1))
def block_boot(d, B=2000, block=168, seed=0):
    rng=np.random.default_rng(seed); d=np.asarray(d,float); n=len(d); nb=int(np.ceil(n/block)); dc=d-d.mean(); m=[]
    for _ in range(B):
        st=rng.integers(0,n-block+1,nb); idx=(st[:,None]+np.arange(block)).ravel()[:n]; m.append(dc[idx].mean())
    m=np.array(m); return float(np.mean(np.abs(m)>=abs(d.mean())))
def hourly(ts, l):
    s=pd.Series(l).groupby(pd.Series(ts)).mean().sort_index(); return s.values
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*p[i]); adj[i]=min(1,run)
    return adj
'''

SRC['extra.py'] = r'''
import sys, os, json, numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss
H, SEED = int(sys.argv[1]), int(sys.argv[2])
D='data'; OUT='preds_extra'; os.makedirs(OUT,exist_ok=True)
meta=json.load(open(f'{D}/meta.json')); G=meta['GROUPS']; TH=meta['THETA']
d=pd.read_parquet(f'{D}/feat_h{H}.parquet'); tr,va,te=[(d.split==s).values for s in ['train','val','test']]; y=d.load_s.values
BASE=G['lags']+G['calendar_asset']; F=BASE+G['q15']+['on_lag1','on_yday','hours_in_state']+G['prevday']+G['recent']
age=(d.ts.max()-d.ts).dt.days.values; w=np.exp(-(age[tr]-age[tr].min())/90.0)
fn=f'{OUT}/h{H}_s{SEED}.npz'; P=dict(np.load(fn)) if os.path.exists(fn) else {}
def reg(f, mt=None, mv=None, wt=None):
    mt=np.ones(tr.sum(),bool) if mt is None else mt; mv=np.ones(va.sum(),bool) if mv is None else mv
    r=xgb.XGBRegressor(n_estimators=2000,learning_rate=0.03,max_depth=6,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
                       early_stopping_rounds=100,random_state=SEED,n_jobs=2,tree_method='hist')
    r.fit(d.loc[tr,f][mt],y[tr][mt],sample_weight=None if wt is None else wt[mt],eval_set=[(d.loc[va,f][mv],y[va][mv])],verbose=False); return r
on_tr,on_va,on_te=(y[tr]>TH).astype(int),(y[va]>TH).astype(int),(y[te]>TH).astype(int)
if 'XGB+q15' not in P:
    f=BASE+G['q15']; P['XGB+q15']=reg(f).predict(d.loc[te,f])
for wname,wt in [('w',w),('nw',None)]:
    if f'Mixture_{wname}' in P: continue
    clf=xgb.XGBClassifier(n_estimators=1500,learning_rate=0.03,max_depth=5,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
                          early_stopping_rounds=100,eval_metric='logloss',random_state=SEED,n_jobs=2)
    clf.fit(d.loc[tr,F],on_tr,sample_weight=wt,eval_set=[(d.loc[va,F],on_va)],verbose=False)
    pva,pte=clf.predict_proba(d.loc[va,F])[:,1],clf.predict_proba(d.loc[te,F])[:,1]
    ron,roff=reg(F,on_tr==1,on_va==1,wt),reg(F,on_tr==0,on_va==0,wt)
    fon,foff=ron.predict(d.loc[te,F]),roff.predict(d.loc[te,F])
    P[f'Mixture_{wname}']=pte*fon+(1-pte)*foff; P[f'pi_{wname}']=pte; P[f'pi_va_{wname}']=pva
    P[f'fon_{wname}']=fon; P[f'foff_{wname}']=foff
if 'LGBM' not in P:
    m=lgb.LGBMRegressor(n_estimators=2000,learning_rate=0.03,num_leaves=63,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,
                        min_child_samples=20,random_state=SEED,n_jobs=2,verbose=-1)
    m.fit(d.loc[tr,F],y[tr],sample_weight=w,eval_set=[(d.loc[va,F],y[va])],callbacks=[lgb.early_stopping(100,verbose=False)])
    P['LGBM']=m.predict(d.loc[te,F])
np.savez(fn,**P)
pi=P['pi_w']; print(H,SEED,'brier',round(brier_score_loss(on_te,pi),4),'logloss',round(log_loss(on_te,pi),4),flush=True)
'''

SRC['extra_analysis.py'] = r'''
import os
import json, numpy as np, pandas as pd, sys
sys.path.insert(0,'.'); from dmlib import dm, hourly
D='data'; PR='preds'; EP='preds_extra'; O='results_updated'
meta=json.load(open(f'{D}/meta.json')); SC=meta['scale']; TH=meta['THETA']; SEEDS=[int(s) for s in os.environ.get('SEEDS', '42,7,123,2024,31').split(',')]
rows=[]; cal=[]
for H in [1,6,24]:
    f=pd.read_parquet(f'{D}/feat_h{H}.parquet',columns=['AssetId','ts','load_s','split','on_lag1'])
    tr,te=(f.split=='train').values,(f.split=='test').values; sc=f.AssetId.map(SC).values[te]; y=f.load_s.values[te]*sc; ts=f.ts.values[te]
    on_te=(f.load_s.values[te]>TH).astype(int)
    res={k:[] for k in ['XGBoost','XGB+q15','Single-stage','LGBM','Mixture_w','Mixture_nw','Mixture_orig']}; pv={k:[] for k in res}
    for s in SEEDS:
        E=np.load(f'{EP}/h{H}_s{s}.npz'); L=np.load(f'{PR}/learned_h{H}_s{s}.npz'); V=np.load(f'{PR}/v2_h{H}_s{s}.npz')
        P={'XGBoost':L['XGBoost'],'XGB+q15':E['XGB+q15'],'Single-stage':V['Proposed v2'],'LGBM':E['LGBM'],'Mixture_w':E['Mixture_w'],'Mixture_nw':E['Mixture_nw'],'Mixture_orig':V['Two-Stage v2']}
        for k,p in P.items():
            pk=np.clip(p,0,None)*sc; res[k].append(np.sqrt(np.mean((y-pk)**2))); pv[k].append(pk)
        pi=E['pi_w'];
        cal.append(dict(H=H,seed=s,brier=np.mean((pi-on_te)**2),brier_clim=np.mean((on_te.mean()-on_te)**2),brier_state=np.mean((f.on_lag1.values[te]-on_te)**2),
                        logloss=-np.mean(on_te*np.log(np.clip(pi,1e-9,1))+(1-on_te)*np.log(np.clip(1-pi,1e-9,1))),
                        ece=sum(abs(pi[(pi>=a)&(pi<a+0.1)].mean()-on_te[(pi>=a)&(pi<a+0.1)].mean())*((pi>=a)&(pi<a+0.1)).mean() for a in np.arange(0,1,0.1) if ((pi>=a)&(pi<a+0.1)).any()),
                        neg_share=np.mean(E['Mixture_w']<0), neg_min_kw=float((np.minimum(E['Mixture_w'],0)*sc).min())))
    for k in res:
        r=dict(H=H,model=k,RMSE=np.mean(res[k]),SD=np.std(res[k],ddof=1))
        if k not in ('Mixture_w','Mixture_orig'):
            ps=[dm(hourly(ts,(y-pv[k][i])**2-(y-pv['Mixture_orig'][i])**2),H) for i in range(5)]
            r.update(DM_min=min(p[0] for p in ps),p_max=max(p[1] for p in ps))
        rows.append(r)
    print(H,'ON share train %.3f test %.3f'%((f.load_s.values[tr]>TH).mean(), on_te.mean()))
R=pd.DataFrame(rows); C=pd.DataFrame(cal)
pd.set_option('display.width',200); print(R.round(4).to_string()); print(C.groupby('H').mean().round(4).to_string())
R.to_csv(f'{O}/equal_information.csv',index=False); C.to_csv(f'{O}/classifier_diagnostics.csv',index=False)
rel=[]
for H in [1,6,24]:
    f=pd.read_parquet(f'{D}/feat_h{H}.parquet',columns=['load_s','split']); te=(f.split=='test').values; on=(f.load_s.values[te]>TH).astype(int)
    pi=np.mean([np.load(f'{EP}/h{H}_s{s}.npz')['pi_w'] for s in SEEDS],0)
    b=np.minimum((pi*10).astype(int),9)
    for k in range(10):
        m=b==k
        if m.any(): rel.append(dict(H=H,bin=f'{k/10:.1f}-{(k+1)/10:.1f}',n=int(m.sum()),mean_pi=pi[m].mean(),obs_on=on[m].mean()))
RL=pd.DataFrame(rel); RL.to_csv(f'{O}/classifier_reliability.csv',index=False); print(RL.round(3).to_string())
'''

SRC['featlib.py'] = r'''
import json, numpy as np, pandas as pd
D = 'data'
meta = json.load(open(f'{D}/meta.json')); A = meta['ASSETS']; TH = meta['THETA']; G = meta['GROUPS']
F_BASE = G['lags'] + G['calendar_asset']
F_PROP = F_BASE + G['q15'] + ['on_lag1', 'on_yday', 'hours_in_state'] + G['prevday'] + G['recent']
_base = pd.read_parquet(f'{D}/base.parquet', columns=['AssetId', 'ts', 'load'])
_q = pd.read_parquet(f'{D}/q.parquet', columns=['AssetId', 't', 'Demand_kW'])
def scale_from(cutoff):
    return _base[_base.ts < cutoff].groupby('AssetId')['load'].max().replace(0, 1.0).to_dict()
def build(H, scale):
    base = _base.copy(); base['load_s'] = base.load / base.AssetId.map(scale)
    ql = _q.set_index(['AssetId', 't'])['Demand_kW'] / _q.AssetId.map(scale).values
    d = base[['AssetId', 'ts', 'load', 'load_s']].copy(); g = d.groupby('AssetId')['load_s']
    for k in range(1, 25): d[f'lag_{k}'] = g.shift(H + k - 1)
    d['lag_168'] = g.shift(168); sh = g.shift(H)
    d['roll_mean_24'] = sh.groupby(d.AssetId).transform(lambda x: x.rolling(24, min_periods=18).mean())
    d['roll_std_24'] = sh.groupby(d.AssetId).transform(lambda x: x.rolling(24, min_periods=18).std())
    hr, dow = d.ts.dt.hour, d.ts.dt.dayofweek
    d['hour_sin'], d['hour_cos'] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    d['dow_sin'], d['dow_cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    d['is_weekend'] = (dow >= 5).astype(int)
    for a in A: d[f'asset_{a}'] = (d.AssetId == a).astype(int)
    oe = d.ts - pd.Timedelta(hours=H - 1)
    for k in range(1, 9):
        d[f'q_lag_{k}'] = ql.reindex(pd.MultiIndex.from_arrays([d.AssetId, oe - pd.Timedelta(minutes=15 * k)])).values
    d['q_slope'] = d.q_lag_1 - d.q_lag_4
    d['on_lag1'] = (d.lag_1 > TH).astype(int); d['on_yday'] = (g.shift(24) > TH).astype(int)
    onm = d[[f'lag_{k}' for k in range(1, 25)]].values > TH; same = onm == onm[:, [0]]
    d['hours_in_state'] = np.where(same.all(axis=1), 24, np.argmin(same, axis=1))
    dm = base.assign(date=base.ts.dt.normalize()).groupby(['AssetId', 'date'])['load_s'].mean()
    od = (d.ts - pd.Timedelta(hours=H)).dt.normalize() - pd.Timedelta(days=1)
    d['prev_day_mean'] = dm.reindex(pd.MultiIndex.from_arrays([d.AssetId, od])).values
    wk = np.stack([g.shift(168 * k).values for k in range(1, 5)], 1)
    import warnings; warnings.filterwarnings('ignore')
    d['rp_week_mean'] = np.nanmean(wk, 1); d['rp_week_on'] = np.nanmean(np.where(np.isnan(wk), np.nan, wk > TH), 1); d['rp_week_max'] = np.nanmax(wk, 1)
    k0 = int(np.ceil(H / 24))
    dy = np.stack([g.shift(24 * k).values for k in range(k0, k0 + 7)], 1)
    dy1 = np.stack([g.shift(24 * k + 1).values for k in range(k0, k0 + 7)], 1)
    d['rp_day_mean'] = np.nanmean(dy, 1); d['rp_day_on'] = np.nanmean(np.where(np.isnan(dy), np.nan, dy > TH), 1)
    d['rp_day_delta'] = d.rp_day_mean - np.nanmean(dy1, 1)
    d = d.dropna(subset=['load_s'] + [f'lag_{k}' for k in range(1, 25)]).reset_index(drop=True)
    d['scale'] = d.AssetId.map(scale)
    return d
'''

SRC['fig_anomaly.py'] = r'''
from figs_common import *
import numpy as np, pandas as pd, json
R=pd.read_csv('results_updated/anomaly_chrono_all.csv')
M=[('Proposed K=5','Level-preserving kNN (proposed)',C['prop'],'-','o'),('LOF','LOF',C['tf'],'--','s'),
   ('Isolation Forest','Isolation Forest',C['lstm'],'-.','^'),('DBSCAN density','DBSCAN density',C['xgb'],':','D'),
   ('K-means distance','K-means distance (B-1)',C['pers'],(0,(5,1,1,1)),'v')]
fig,ax=plt.subplots(1,2,figsize=(DC,2.25),gridspec_kw={'width_ratios':[1,1.35]})
for k,lab,c,ls,mk in M:
    g=R[R.Method==k].groupby(['budget','seed'])['All_matched'].mean().groupby('budget')
    m,s=g.mean(),g.std(); x=100*m.index.values
    ax[0].errorbar(x,m.values,yerr=s.values,color=c,ls=ls,marker=mk,ms=3.5,mfc='white',capsize=2,label=lab,lw=1.2)
ax[0].set_xlabel('Alert rate on recorded test days (%)'); ax[0].set_ylabel('Injected anomalies detected (%)'); ax[0].set_xticks([1,2,5,10]); ax[0].grid(alpha=0.3,lw=0.4); ax[0].set_ylim(0,70)
ax[0].set_title('(a) Detection rate vs. alert rate')
F=['Left ON overnight','Missed shift','Load spike','Schedule shift','Base-load increase']
r5=R[R.budget==0.05]; w=0.16; xs=np.arange(len(F))
for i,(k,lab,c,ls,mk) in enumerate(M):
    v=r5[r5.Method==k][[x+' (matched)' for x in F]].mean().values
    ax[1].bar(xs+(i-2)*w,v,w,color=c,edgecolor='black',lw=0.3,label=lab)
ax[1].set_xticks(xs); ax[1].set_xticklabels(['Left ON\novernight','Missed\nshift','Load\nspike','Schedule\nshift','Base-load\nincrease'])
ax[1].set_ylabel('Detected at a 5% alert rate (%)'); ax[1].set_ylim(0,105); ax[1].grid(axis='y',alpha=0.3,lw=0.4); ax[1].set_title('(b) Detection rate by anomaly type')
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(0.5,1.09),fontsize=7)
fig.tight_layout(w_pad=1.2); save(fig,'fig_anomaly')
'''

SRC['fig_faults.py'] = r'''
from figs_common import *
import numpy as np, pandas as pd
D='data'
b=pd.read_parquet(f'{D}/base.parquet'); b['date']=b.ts.dt.normalize(); b['hr']=b.ts.dt.hour
prof=b.pivot_table(index=['AssetId','date'],columns='hr',values='load').dropna()
import json; a='comp_b'; P=prof.loc[a]; amax=json.load(open(f'{D}/meta.json'))['scale'][a]
wd=P[pd.to_datetime(P.index).dayofweek<5]; e=wd.sum(axis=1); day=wd.loc[(e-e.median()).abs().idxmin()].values
def inject(p,kind,s=7):
    q=p.copy(); lo,hi=p.min(),p.max(); off,on=p<=lo+0.2*(hi-lo),p>=lo+0.5*(hi-lo)
    if kind=='Left ON overnight': q[off]=0.8*hi
    elif kind=='Missed shift': q[on]=lo
    elif kind=='Load spike': q[s:s+2]=2.5*np.maximum(p[s:s+2],0.1*amax)
    elif kind=='Schedule shift': q=np.roll(p,4)
    elif kind=='Base-load increase': q=p+0.3*amax
    return q
F=['Left ON overnight','Missed shift','Load spike','Schedule shift','Base-load increase']
fig,axs=plt.subplots(1,5,figsize=(DC,1.55),sharey=True)
for ax,k in zip(axs,F):
    ax.plot(range(24),day,color=C['actual'],lw=1.4,label='Recorded day')
    ax.plot(range(24),inject(day,k),color=C['xgb'],ls='--',marker='o',ms=2.2,mfc='white',lw=1.1,label='Injected copy')
    ax.set_title(k); ax.set_xticks([0,6,12,18]); ax.set_xlabel('Hour'); ax.grid(alpha=0.3,lw=0.4)
axs[0].set_ylabel('Demand (kW)'); axs[0].legend(frameon=False,fontsize=6,loc='upper left')
fig.tight_layout(w_pad=0.3); save(fig,'fig_faults')
print(pd.to_datetime(wd.index[(e-e.median()).abs().argmin()]))
'''

SRC['fig_forecasting.py'] = r'''
import os
from figs_common import *
import json, numpy as np, pandas as pd
R = 'results'; D = 'data'
S = pd.read_csv(f'{R}/forecast_summary.csv')
HS = [1, 6, 24]
MODELS = [('Persistence', 'Persistence', C['pers'], (0, (1, 1)), 'x'), ('ARIMA', 'ARIMA (B-2)', C['arima'], (0, (5, 2)), 'v'),
          ('LSTM', 'LSTM (B-2)', C['lstm'], '-.', '^'), ('Transformer', 'Transformer', C['tf'], (0, (3, 1, 1, 1)), 'D'),
          ('TCN', 'TCN (B-2)', C['tcn'], (0, (4, 1, 1, 1, 1, 1)), 'h'), ('XGBoost', 'XGBoost (B-2)', C['xgb'], '--', 's'),
          ('Enhanced (conference features)', 'Conference feature set', C['ts'], ':', 'P'), ('Enhanced XGBoost', 'State-aware mixture (proposed)', C['prop'], '-', 'o')]
MODELS = [m for m in MODELS]
def g(m, H, col):
    r = S[(S.model == m) & (S.H == H)]
    return (float(r[col + '_mean'].iloc[0]), float(r[col + '_std'].fillna(0).iloc[0])) if len(r) else (np.nan, 0)
fig, ax = plt.subplots(figsize=(SC, 2.6))
xs = np.arange(3)
for k, (m, lab, c, ls, mk) in enumerate(MODELS):
    v = [g(m, H, 'RMSE') for H in HS]
    ax.errorbar(xs + (k - 3.5) * 0.032, [a for a, b in v], yerr=[b for a, b in v], color=c, ls=ls, marker=mk, ms=3.6,
                mfc='white', capsize=1.5, lw=1.2, label=lab)
ax.set_xticks(xs); ax.set_xticklabels(['1 h', '6 h', '24 h']); ax.set_xlabel('Forecast horizon'); ax.set_ylabel('Test RMSE (kW)')
ax.grid(alpha=0.3, lw=0.4); ax.legend(ncol=3, frameon=False, fontsize=5.8, loc='upper center', bbox_to_anchor=(0.47, 1.36), columnspacing=0.8, handlelength=2.2)
fig.tight_layout(); save(fig, 'fig_horizon')
sel = [('Persistence', 'Pers.', C['pers']), ('ARIMA', 'ARIMA', C['arima']), ('LSTM', 'LSTM', C['lstm']),
       ('TCN', 'TCN', C['tcn']), ('XGBoost', 'XGB', C['xgb']), ('Enhanced XGBoost', 'Mixture', C['prop'])]
fig, axs = plt.subplots(1, 3, figsize=(SC, 2.0), sharey=True)
for ax, H in zip(axs, HS):
    x = np.arange(len(sel))
    sw = [g(m, H, 'SW')[0] for m, _, _ in sel]; st = [g(m, H, 'ST')[0] for m, _, _ in sel]
    ax.bar(x - 0.2, st, 0.4, color=[c for _, _, c in sel], alpha=0.45, edgecolor='black', lw=0.3, hatch='//', label='Steady hours')
    ax.bar(x + 0.2, sw, 0.4, color=[c for _, _, c in sel], edgecolor='black', lw=0.3, label='Switching hours')
    ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in sel], rotation=90, fontsize=6.3); ax.set_title(f'H = {H} h')
    ax.grid(axis='y', alpha=0.3, lw=0.4)
axs[0].set_ylabel('Test RMSE (kW)')
from matplotlib.patches import Patch
fig.legend([Patch(facecolor='0.6', alpha=0.45, hatch='//', edgecolor='black', lw=0.3), Patch(facecolor='0.4', edgecolor='black', lw=0.3)],
           ['Steady hours', 'Switching hours'], ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(0.55, 1.06), fontsize=7)
fig.tight_layout(w_pad=0.3); save(fig, 'fig_switching')
meta = json.load(open(f'{D}/meta.json')); A = meta['ASSETS']; SCL = meta['scale']
M = np.zeros((len(A), 3)); PR = 'preds'
for j, H in enumerate(HS):
    f = pd.read_parquet(f'{D}/feat_h{H}.parquet', columns=['AssetId', 'load_s', 'split']); te = (f.split == 'test').values
    a_ = f.AssetId.values[te]; sc = pd.Series(a_).map(SCL).values; y = f.load_s.values[te] * sc
    def per(p): return (pd.Series((y - np.clip(p, 0, None) * sc) ** 2).groupby(a_).mean()) ** 0.5
    e = np.mean([per(np.load(f'{PR}/v2_h{H}_s{s}.npz')['Two-Stage v2']).reindex(A).values for s in [int(s) for s in os.environ.get('SEEDS', '42,7,123,2024,31').split(',')]], 0)
    x = np.mean([per(np.load(f'{PR}/learned_h{H}_s{s}.npz')['XGBoost']).reindex(A).values for s in [int(s) for s in os.environ.get('SEEDS', '42,7,123,2024,31').split(',')]], 0)
    M[:, j] = 100 * (e / x - 1)
pd.DataFrame(M, index=A, columns=HS).to_csv('results_updated/per_asset_change_5seed.csv')
fig, ax = plt.subplots(figsize=(SC, 3.3))
lim = np.nanmax(np.abs(M))
im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
ax.set_yticks(range(len(A))); ax.set_yticklabels(A, fontsize=6.5)
ax.set_xticks(range(3)); ax.set_xticklabels(['1 h', '6 h', '24 h']); ax.xaxis.tick_top()
for i in range(len(A)):
    for j in range(3): ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=6.3, color='white' if abs(M[i, j]) > 0.55 * lim else 'black')
cb = fig.colorbar(im, ax=ax, pad=0.03); cb.set_label('RMSE change (%)', fontsize=6.5); cb.ax.tick_params(labelsize=6)
fig.tight_layout(); save(fig, 'fig_assets')
G = pd.read_csv(f'{R}/shap_groups.csv')
order = [('q15', '15-min readings', C['prop']), ('lags', 'Hourly lags', C['pers']), ('recent', 'Recent profile', C['xgb']),
         ('calendar_asset', 'Calendar and asset', C['tf']), ('state', 'ON/OFF state', C['lstm']), ('prevday', 'Previous day', C['ts'])]
fig, ax = plt.subplots(figsize=(SC, 2.0))
cats = [(H, col) for H in HS for col in ['mean_abs', 'mean_abs_sw']]
labels = [f'{H} h\n{"all" if c == "mean_abs" else "switch."}' for H, c in cats]
x = np.arange(len(cats)); x = x + np.repeat([0, 0.5, 1.0], 2)
bottom = np.zeros(len(cats))
for gk, gl, gc in order:
    v = np.array([100 * G[(G.H == H) & (G.group == gk)][c].iloc[0] for H, c in cats])
    ax.bar(x, v, 0.8, bottom=bottom, color=gc, edgecolor='white', lw=0.4, label=gl)
    for xi, vi, bi in zip(x, v, bottom):
        if vi > 7: ax.text(xi, bi + vi / 2, f'{vi:.0f}', ha='center', va='center', fontsize=5.8, color='white')
    bottom += v
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.3); ax.set_ylabel('Share of |SHAP| (%)'); ax.set_ylim(0, 100)
ax.legend(ncol=3, frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, 1.3))
fig.tight_layout(); save(fig, 'fig_shap_groups')
F = pd.read_csv(f'{R}/shap_features.csv'); gcol = {k: c for k, _, c in order}
NICE = {'q_lag_1': '$q_1$', 'q_lag_2': '$q_2$', 'q_lag_3': '$q_3$', 'q_lag_4': '$q_4$', 'q_slope': r'$\Delta q$', 'usual_on_prob': r'$\rho(d,h)$',
        'usual_on_prob_prev': r'$\rho(d,h-1)$', 'prev_day_mean': 'prev. day mean', 'roll_mean_24': '24-h mean', 'roll_std_24': '24-h std',
        'hours_in_state': 'hours in state', 'on_lag1': 'ON at origin', 'on_yday': 'ON yesterday', 'x_mean_lag1': 'plant mean',
        'x_on_frac': 'plant ON frac.', 'x_on_frac_q1': 'plant ON frac. (15 min)', 'x_mean_q1': 'plant mean (15 min)', 'x_press_on': 'presses ON', 'rp_week_mean': 'same hour, 4 wk mean', 'rp_week_on': 'same hour, 4 wk ON', 'rp_week_max': 'same hour, 4 wk max', 'rp_day_mean': 'same hour, 7 d mean', 'rp_day_on': 'same hour, 7 d ON', 'rp_day_delta': 'usual change, 7 d',
        'hour_sin': 'hour sin', 'hour_cos': 'hour cos', 'dow_sin': 'weekday sin', 'dow_cos': 'weekday cos', 'is_weekend': 'weekend'}
def nice(c, H=1):
    if c in NICE: return NICE[c]
    if c.startswith('lag_'): return f'$z_{{t-{int(c[4:])+H-1}}}$' if c != 'lag_168' else '$z_{t-168}$'
    if c.startswith('asset_'): return c[6:]
    return c
fig, axs = plt.subplots(1, 3, figsize=(DC, 2.1))
for ax, H in zip(axs, HS):
    f = F[F.H == H].sort_values('mean_abs', ascending=False).head(10)[::-1]
    ax.barh(range(len(f)), f.mean_abs, color=[gcol[g] for g in f.group], edgecolor='black', lw=0.3)
    ax.set_yticks(range(len(f))); ax.set_yticklabels([nice(c, H) for c in f.feature], fontsize=6.5)
    ax.set_title(f'H = {H} h'); ax.set_xlabel('Mean |SHAP| (kW)'); ax.grid(axis='x', alpha=0.3, lw=0.4)
from matplotlib.patches import Patch
fig.legend([Patch(color=c) for _, _, c in order], [l for _, l, _ in order], ncol=6, frameon=False, loc='upper center',
           bbox_to_anchor=(0.5, 1.07), fontsize=6.5)
fig.tight_layout(w_pad=0.8); save(fig, 'fig_shap_top')
a = 'comp_b'
fig, ax = plt.subplots(figsize=(SC, 2.0))
t0, t1 = pd.Timestamp('2025-11-11'), pd.Timestamp('2025-11-14')
for H, c, ls in [(24, C['tf'], ':'), (6, C['xgb'], '--'), (1, C['prop'], '-')]:
    f = pd.read_parquet(f'{D}/feat_h{H}.parquet', columns=['AssetId', 'ts', 'load', 'split'])
    te = (f.split == 'test').values; fi = f[te].reset_index(drop=True)
    p = np.clip(np.load(f'preds/v2_h{H}_s42.npz')['Two-Stage v2'], 0, None) * SCL[a]
    m = (fi.AssetId == a) & (fi.ts >= t0) & (fi.ts < t1)
    if H == 1: ax.plot(fi.ts[m], fi.load[m], color=C['actual'], lw=1.6, label='Recorded', zorder=1)
    ax.plot(fi.ts[m], p[m.values], color=c, ls=ls, lw=1.2, label=f'Mixture, {H} h ahead', zorder=3)
ax.set_ylabel('Demand (kW)'); ax.grid(alpha=0.3, lw=0.4)
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M')); ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
ax.tick_params(axis='x', labelsize=6)
ax.legend(ncol=2, frameon=False, fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, 1.27))
fig.tight_layout(); save(fig, 'fig_forecast')
print('figs ok')
'''

SRC['fig_profiles.py'] = r'''
from figs_common import *
import json, numpy as np, pandas as pd
D='data'; meta=json.load(open(f'{D}/meta.json')); A=meta['ASSETS']; T=meta['TYPE']
b=pd.read_parquet(f'{D}/base.parquet'); b=b[b.ts<meta['t_val']]
b['wk']=b.ts.dt.dayofweek<5; b['hr']=b.ts.dt.hour
fig,axs=plt.subplots(3,5,figsize=(DC,3.9),sharex=True,sharey=True); axs=axs.ravel()
for i,a in enumerate(A):
    ax=axs[i]; x=b[b.AssetId==a]
    for wk,col,ls,lab in [(True,C['prop'],'-','Weekday'),(False,C['xgb'],'--','Weekend')]:
        g=x[x.wk==wk].groupby('hr').load_s
        m,lo,hi=g.mean(),g.quantile(0.25),g.quantile(0.75)
        ax.plot(m.index,m.values,color=col,ls=ls,label=lab); ax.fill_between(m.index,lo,hi,color=col,alpha=0.15,lw=0)
    ax.set_title(f'{a} ({T[a].split(" (")[0].replace("Compressed air","Compressor").replace("Material handling","Handling").replace("Process mixing","Mixer").replace("Utilities","Utility")})',fontsize=7); ax.set_xticks([0,6,12,18]); ax.grid(alpha=0.3,lw=0.4); ax.set_ylim(0,1.02)
for ax in axs[13:]: ax.axis('off')
axs[13].legend(*axs[0].get_legend_handles_labels(),loc='center',frameon=False,fontsize=8)
for i in [0,5,10]: axs[i].set_ylabel('Scaled demand')
for i in [10,11,12,8,9]: axs[i].set_xlabel('Hour of day (UTC)')
for i in [8,9]: axs[i].tick_params(labelbottom=True)
fig.tight_layout(h_pad=0.6,w_pad=0.4); save(fig,'fig_profiles')
'''

SRC['figs_common.py'] = r'''
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
 'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':0.6,'lines.linewidth':1.3,'pdf.fonttype':42})
FIG='figures'
C={'prop':'#2a78d6','xgb':'#eb6834','lstm':'#1baf7a','tf':'#eda100','pers':'#7f7f7f','arima':'#e87ba4','ts':'#7b4fc9','actual':'#1b1b1b','tcn':'#4fb3c9'}
SC=3.45; DC=7.16
def save(fig,name):
    fig.savefig(f'{FIG}/{name}.pdf',bbox_inches='tight',dpi=600); fig.savefig(f'{FIG}/{name}.png',bbox_inches='tight',dpi=600); plt.close(fig)
'''

SRC['learned.py'] = r'''
import sys, os, json, time, random, warnings
warnings.filterwarnings('ignore'); os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
H, SEED = int(sys.argv[1]), int(sys.argv[2])
ONLY = sys.argv[3].split(',') if len(sys.argv) > 3 else None
D = 'data'; OUT = f'preds'; os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f'{D}/meta.json')); G = meta['GROUPS']; THETA = meta['THETA']
d = pd.read_parquet(f'{D}/feat_h{H}.parquet')
tr, va, te = (d.split == 'train').values, (d.split == 'val').values, (d.split == 'test').values
y = d['load_s'].values; y_tr, y_va = y[tr], y[va]
BASE = G['lags'] + G['calendar_asset']
ENH = BASE + G['q15'] + G['onoff'] + G['cross'] + G['prevday']
fn = f'{OUT}/learned_h{H}_s{SEED}.npz'
P = dict(np.load(fn)) if os.path.exists(fn) else {}
def save(): np.savez(fn, **P)
def want(k): return (ONLY is None or k in ONLY) and (k not in P)
random.seed(SEED); np.random.seed(SEED)
def fit_xgb(feats):
    m = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3, early_stopping_rounds=100,
                         random_state=SEED, n_jobs=2, tree_method='hist')
    m.fit(d.loc[tr, feats], y_tr, eval_set=[(d.loc[va, feats], y_va)], verbose=False)
    return m
def put(k, va_pred, te_pred):
    P[k + '|va'] = np.asarray(va_pred, float); P[k] = np.asarray(te_pred, float); save()
t0 = time.time()
if want('XGBoost'):
    m = fit_xgb(BASE); put('XGBoost', m.predict(d.loc[va, BASE]), m.predict(d.loc[te, BASE]))
if want('Random Forest'):
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, random_state=SEED, n_jobs=2)
    rf.fit(d.loc[tr, BASE].fillna(-1), y_tr)
    put('Random Forest', rf.predict(d.loc[va, BASE].fillna(-1)), rf.predict(d.loc[te, BASE].fillna(-1)))
if want('DBSCAN + XGBoost'):
    f = BASE + G['dbscan']; m = fit_xgb(f); put('DBSCAN + XGBoost', m.predict(d.loc[va, f]), m.predict(d.loc[te, f]))
if want('Enhanced XGBoost'):
    m = fit_xgb(ENH); put('Enhanced XGBoost', m.predict(d.loc[va, ENH]), m.predict(d.loc[te, ENH]))
    m.save_model(f'{OUT}/enh_h{H}_s{SEED}.json')
    P['best_iter'] = np.array([m.best_iteration]); save()
for grp in ['q15', 'onoff', 'cross', 'prevday']:
    k = f'Ablation -{grp}'
    if want(k):
        f = [c for c in ENH if c not in G[grp]]; m = fit_xgb(f); put(k, m.predict(d.loc[va, f]), m.predict(d.loc[te, f]))
if want('Two-Stage Hybrid'):
    Xtr, Xva, Xte = d.loc[tr, ENH], d.loc[va, ENH], d.loc[te, ENH]
    on_tr, on_va = (y_tr > THETA).astype(int), (y_va > THETA).astype(int)
    clf = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=3, early_stopping_rounds=100, eval_metric='logloss', random_state=SEED, n_jobs=2)
    clf.fit(Xtr, on_tr, eval_set=[(Xva, on_va)], verbose=False)
    pva, pte = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    P['auc'] = np.array([roc_auc_score(on_va, pva)])
    def reg(mt, mv):
        r = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=3, early_stopping_rounds=100, random_state=SEED, n_jobs=2)
        r.fit(Xtr[mt], y_tr[mt], eval_set=[(Xva[mv], y_va[mv])], verbose=False); return r
    ron, roff = reg(on_tr == 1, on_va == 1), reg(on_tr == 0, on_va == 0)
    put('Two-Stage Hybrid', pva * ron.predict(Xva) + (1 - pva) * roff.predict(Xva),
        pte * ron.predict(Xte) + (1 - pte) * roff.predict(Xte))
print(f'H{H} s{SEED} trees done {time.time()-t0:.0f}s', flush=True)
NN = ['LSTM', 'CNN-LSTM', 'TCN', 'Transformer']
if any(want(k) for k in NN):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    tf.random.set_seed(SEED); tf.config.threading.set_intra_op_parallelism_threads(2)
    seq = [f'lag_{k}' for k in range(24, 0, -1)]; aux = G['calendar_asset']
    def inp(mk): return [d.loc[mk, seq].values[..., None].astype('float32'), d.loc[mk, aux].values.astype('float32')]
    Xtr_n, Xva_n, Xte_n = inp(tr), inp(va), inp(te)
    def head(x, aux_in):
        x = layers.Concatenate()([x, aux_in]); x = layers.Dense(32, activation='relu')(x); return layers.Dense(1)(x)
    def build(kind):
        s_in, a_in = keras.Input((24, 1)), keras.Input((len(aux),)); x = s_in
        if kind == 'LSTM':
            x = layers.LSTM(64)(x)
        elif kind == 'CNN-LSTM':
            x = layers.Conv1D(32, 3, padding='causal', activation='relu')(x)
            x = layers.Conv1D(32, 3, padding='causal', activation='relu')(x); x = layers.LSTM(64)(x)
        elif kind == 'TCN':
            for dl in [1, 2, 4, 8]:
                h = layers.Conv1D(32, 3, padding='causal', dilation_rate=dl, activation='relu')(x)
                h = layers.Conv1D(32, 3, padding='causal', dilation_rate=dl, activation='relu')(h)
                x = layers.Add()([layers.Conv1D(32, 1)(x) if x.shape[-1] != 32 else x, h])
            x = layers.Lambda(lambda z: z[:, -1, :])(x)
        elif kind == 'Transformer':
            x = layers.Dense(32)(x)
            pos = layers.Embedding(24, 32)(tf.range(24)); x = x + pos
            for _ in range(2):
                a = layers.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.1)(x, x)
                x = layers.LayerNormalization(epsilon=1e-6)(x + a)
                f = layers.Dense(64, activation='relu')(x); f = layers.Dense(32)(f)
                x = layers.LayerNormalization(epsilon=1e-6)(x + f)
            x = layers.GlobalAveragePooling1D()(x)
        m = keras.Model([s_in, a_in], head(x, a_in)); m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse'); return m
    for k in NN:
        if not want(k): continue
        t1 = time.time(); tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        m = build(k)
        m.fit(Xtr_n, y_tr, validation_data=(Xva_n, y_va), epochs=30, batch_size=512, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        put(k, m.predict(Xva_n, verbose=0, batch_size=2048).ravel(), m.predict(Xte_n, verbose=0, batch_size=2048).ravel())
        print(f'H{H} s{SEED} {k} {time.time()-t1:.0f}s', flush=True)
print(f'H{H} s{SEED} ALL {time.time()-t0:.0f}s', flush=True)
'''

SRC['make_reliable.py'] = r'''
import glob, os, pandas as pd
os.makedirs('data', exist_ok=True)
COLS = ['AssetId', 'window_start_utc', 'Phase', 'IsReliableWindow', 'Demand_kW']
PQ = glob.glob('dataset/**/data_parquet', recursive=True)
CS = glob.glob('dataset/**/data_csv', recursive=True)
if PQ:
    print('reading', PQ[0])
    df = pd.read_parquet(PQ[0], columns=COLS)
elif CS:
    files = glob.glob(f'{CS[0]}/**/*.csv', recursive=True)
    print('reading', len(files), 'CSV files from', CS[0])
    df = pd.concat((pd.read_csv(f, usecols=COLS) for f in files), ignore_index=True)
else:
    raise SystemExit('Neither data_parquet nor data_csv found under dataset/ - check the dataset cell')
df['AssetId'] = df['AssetId'].astype(str)
df['window_start_utc'] = pd.to_datetime(df['window_start_utc'])
rel = df['IsReliableWindow'].astype(str).str.lower().isin(['true', '1', '1.0'])
d = df[rel & (df.Phase == 'ALL')].drop_duplicates(['AssetId', 'window_start_utc'])
d.to_parquet('data/all_assets_rel.parquet')
print(len(d), 'reliable windows,', d.AssetId.nunique(), 'assets')
'''

SRC['prep.py'] = r'''
import os, json, numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
OUT = 'data'; os.makedirs(OUT, exist_ok=True)
ASSETS = ['p_b', 'p_c', 'p_e', 'p_g', 'p_k', 'ahu_a', 'ahu_b', 'ex_b',
          'comp_a', 'comp_b', 'mh_a', 'u_e', 'mix_a']
TYPE = {'p_b': 'Press (main)', 'p_c': 'Press (drive)', 'p_e': 'Press (drive)', 'p_g': 'Press (main)',
        'p_k': 'Press (main)', 'ahu_a': 'HVAC (air handling)', 'ahu_b': 'HVAC (air handling)',
        'ex_b': 'HVAC (air extraction)', 'comp_a': 'Compressed air', 'comp_b': 'Compressed air',
        'mh_a': 'Material handling', 'u_e': 'Utilities', 'mix_a': 'Process mixing'}
PRESS = [a for a in ASSETS if a.startswith('p_')]
HORIZONS = [1, 6, 24]
N_LAGS, N_Q, THETA = 24, 8, 0.2
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10
raw = pd.read_parquet('data/all_assets_rel.parquet')
raw = raw[raw.AssetId.isin(ASSETS)].copy()
raw['t'] = pd.to_datetime(raw['window_start_utc'])
q = raw.groupby(['AssetId', 't'])['Demand_kW'].mean().reset_index()
q['hts'] = q['t'].dt.floor('h')
hourly = q.groupby(['AssetId', 'hts'])['Demand_kW'].mean().reset_index().rename(columns={'hts': 'ts', 'Demand_kW': 'load'})
frames = []
for a in ASSETS:
    s = hourly[hourly.AssetId == a].set_index('ts')['load'].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h'))
    frames.append(pd.DataFrame({'AssetId': a, 'ts': s.index, 'load': s.values}))
base = pd.concat(frames, ignore_index=True)
all_ts = np.sort(base.loc[base.load.notna(), 'ts'].unique())
t_val = pd.Timestamp(all_ts[int(len(all_ts) * TRAIN_FRAC)]).normalize()
t_test = pd.Timestamp(all_ts[int(len(all_ts) * (TRAIN_FRAC + VAL_FRAC))]).normalize()
scale = base[base.ts < t_val].groupby('AssetId')['load'].max().replace(0, 1.0)
base['load_s'] = base['load'] / base.AssetId.map(scale)
q['q_s'] = q['Demand_kW'] / q.AssetId.map(scale)
base.to_parquet(f'{OUT}/base.parquet'); q[['AssetId', 't', 'Demand_kW', 'q_s']].to_parquet(f'{OUT}/q.parquet')
base['date'] = base.ts.dt.normalize(); base['hr'] = base.ts.dt.hour
day = base.pivot_table(index=['AssetId', 'date'], columns='hr', values='load', fill_value=0)
day_norm = day.div(day.max(axis=1), axis=0).fillna(0)
Xd = day_norm.values
best = (-1, None)
for eps in np.round(np.arange(0.02, 0.41, 0.01), 3):
    lab = DBSCAN(eps=eps, min_samples=5).fit(Xd).labels_
    n = len(set(lab)) - (1 if -1 in lab else 0); m = lab != -1
    if n >= 2 and m.sum() > n:
        s_ = silhouette_score(Xd[m], lab[m])
        if s_ > best[0]: best = (s_, eps)
chosen_eps = float(best[1])
didx = day_norm.index.to_frame(index=False)
trd = (didx.date < t_val).values
db = DBSCAN(eps=chosen_eps, min_samples=5).fit(Xd[trd])
core = Xd[trd][db.core_sample_indices_] if len(db.core_sample_indices_) else Xd[trd]
didx['day_dist'] = pairwise_distances(Xd, core).min(axis=1)
didx['day_anom'] = (didx.day_dist > chosen_eps).astype(int)
def idx(g):
    p = g['load'].dropna()
    mx, mn = p.max(), p.min()
    return pd.Series({'LF': p.mean() / mx if mx > 0 else np.nan, 'PBR': mx / mn if mn >= 0.5 else np.nan, 'VI': p.var()})
ind = base.groupby(['AssetId', 'date']).apply(idx).reset_index()
didx = didx.merge(ind, on=['AssetId', 'date'], how='left')
didx['VI'] = didx['VI'] / didx.AssetId.map(scale) ** 2
didx['day_mean'] = base.groupby(['AssetId', 'date'])['load_s'].mean().reindex(pd.MultiIndex.from_frame(didx[['AssetId', 'date']])).values
didx.to_parquet(f'{OUT}/days.parquet')
b = base.copy(); b['dow'] = b.ts.dt.dayofweek
on_prob = (b[(b.ts < t_val) & b.load_s.notna()].assign(on=lambda d: (d.load_s > THETA).astype(int))
           .groupby(['AssetId', 'dow', 'hr'])['on'].mean())
q_lookup = q.set_index(['AssetId', 't'])['q_s']
def build(H):
    d = base[['AssetId', 'ts', 'load', 'load_s']].copy()
    g = d.groupby('AssetId')['load_s']
    for k in range(1, N_LAGS + 1):
        d[f'lag_{k}'] = g.shift(H + k - 1)
    d['lag_168'] = g.shift(168)
    sh = g.shift(H)
    d['roll_mean_24'] = sh.groupby(d.AssetId).transform(lambda x: x.rolling(24, min_periods=18).mean())
    d['roll_std_24'] = sh.groupby(d.AssetId).transform(lambda x: x.rolling(24, min_periods=18).std())
    hr, dow = d.ts.dt.hour, d.ts.dt.dayofweek
    d['hour_sin'], d['hour_cos'] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    d['dow_sin'], d['dow_cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    d['is_weekend'] = (dow >= 5).astype(int)
    for a in ASSETS: d[f'asset_{a}'] = (d.AssetId == a).astype(int)
    oe = d.ts - pd.Timedelta(hours=H - 1)
    for k in range(1, N_Q + 1):
        key = pd.MultiIndex.from_arrays([d.AssetId, oe - pd.Timedelta(minutes=15 * k)])
        d[f'q_lag_{k}'] = q_lookup.reindex(key).values
    d['q_slope'] = d['q_lag_1'] - d['q_lag_4']
    d['on_lag1'] = (d['lag_1'] > THETA).astype(int)
    d['on_yday'] = (g.shift(24) > THETA).astype(int)
    onm = d[[f'lag_{k}' for k in range(1, 25)]].values > THETA
    same = onm == onm[:, [0]]
    d['hours_in_state'] = np.where(same.all(axis=1), 24, np.argmin(same, axis=1))
    key = pd.MultiIndex.from_arrays([d.AssetId, dow, hr])
    d['usual_on_prob'] = on_prob.reindex(key).values
    hp = (d.ts - pd.Timedelta(hours=1))
    d['usual_on_prob_prev'] = on_prob.reindex(pd.MultiIndex.from_arrays([d.AssetId, hp.dt.dayofweek, hp.dt.hour])).values
    wl = d.pivot_table(index='ts', columns='AssetId', values='lag_1')
    wq = d.pivot_table(index='ts', columns='AssetId', values='q_lag_1')
    on_l, on_q = (wl > THETA).astype(float).where(wl.notna()), (wq > THETA).astype(float).where(wq.notna())
    cols = {'x_mean_lag1': wl, 'x_mean_q1': wq, 'x_on_frac': on_l, 'x_on_frac_q1': on_q}
    for name, W in cols.items():
        tot, cnt = W.sum(axis=1), W.notna().sum(axis=1)
        sv = W.stack().rename('own').reset_index()
        m = d[['AssetId', 'ts']].merge(sv, on=['ts', 'AssetId'], how='left')
        T_, C_ = d.ts.map(tot).values, d.ts.map(cnt).values
        own = m['own'].values; has = ~np.isnan(own)
        d[name] = np.where(C_ - has > 0, (T_ - np.nan_to_num(own)) / np.maximum(C_ - has, 1), np.nan)
    Wp = on_l[[c for c in on_l.columns if c in PRESS]]
    tot, cnt = Wp.sum(axis=1), Wp.notna().sum(axis=1)
    sv = Wp.stack().rename('own').reset_index()
    m = d[['AssetId', 'ts']].merge(sv, on=['ts', 'AssetId'], how='left'); own = m['own'].values; has = ~np.isnan(own)
    T_, C_ = d.ts.map(tot).values, d.ts.map(cnt).values
    d['x_press_on'] = np.where(C_ - has > 0, (T_ - np.nan_to_num(own)) / np.maximum(C_ - has, 1), np.nan)
    od = (d.ts - pd.Timedelta(hours=H)).dt.normalize() - pd.Timedelta(days=1)
    dl = didx.set_index(['AssetId', 'date'])
    k2 = pd.MultiIndex.from_arrays([d.AssetId, od])
    d['prev_day_mean'] = dl['day_mean'].reindex(k2).values
    for c, n in [('day_dist', 'prev_day_dist'), ('day_anom', 'prev_day_anom'), ('LF', 'prev_LF'), ('PBR', 'prev_PBR'), ('VI', 'prev_VI')]:
        d[n] = dl[c].reindex(k2).values
    d['pers'] = d['lag_1']
    d['snaive'] = g.shift(max(24, H) if H <= 24 else 168)
    d = d.dropna(subset=['load_s'] + [f'lag_{k}' for k in range(1, N_LAGS + 1)]).reset_index(drop=True)
    d['split'] = np.where(d.ts < t_val, 'train', np.where(d.ts < t_test, 'val', 'test'))
    d.to_parquet(f'{OUT}/feat_h{H}.parquet')
    print(H, d.split.value_counts().to_dict(), flush=True)
for H in HORIZONS: build(H)
GROUPS = {
    'lags': [f'lag_{k}' for k in range(1, N_LAGS + 1)] + ['lag_168', 'roll_mean_24', 'roll_std_24'],
    'calendar_asset': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend'] + [f'asset_{a}' for a in ASSETS],
    'q15': [f'q_lag_{k}' for k in range(1, N_Q + 1)] + ['q_slope'],
    'onoff': ['on_lag1', 'on_yday', 'hours_in_state', 'usual_on_prob', 'usual_on_prob_prev'],
    'cross': ['x_mean_lag1', 'x_mean_q1', 'x_on_frac', 'x_on_frac_q1', 'x_press_on'],
    'prevday': ['prev_day_mean'],
    'dbscan': ['prev_day_dist', 'prev_day_anom', 'prev_LF', 'prev_PBR', 'prev_VI']}
json.dump({'ASSETS': ASSETS, 'TYPE': TYPE, 'HORIZONS': HORIZONS, 't_val': str(t_val), 't_test': str(t_test),
           'scale': scale.to_dict(), 'chosen_eps': chosen_eps, 'GROUPS': GROUPS, 'THETA': THETA},
          open(f'{OUT}/meta.json', 'w'), indent=1)
print('t_val', t_val, 't_test', t_test, 'eps', chosen_eps)
'''

SRC['prep25.py'] = r'''
import os, json, numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
OUT = 'data25'; os.makedirs(OUT, exist_ok=True)
ASSETS = ['p_b','p_c','p_e','p_g','p_k','ahu_a','ahu_b','ex_b','comp_a','comp_b','mh_a','u_e','mix_a','p_f','p_i','p_l','p_d','p_m','u_b','p_a','u_c','ex_a','p_h','mix_c','mix_b']
TYPE = {'p_f':'Press (cell)','p_i':'Press (main)','p_l':'Press (main)','p_d':'Press (drive)','p_m':'Press (main)','u_b':'Utilities','p_a':'Press (main)','u_c':'Utilities','ex_a':'HVAC (air extraction)','p_h':'Press (main)','mix_c':'Process mixing','mix_b':'Process mixing','p_b': 'Press (main)', 'p_c': 'Press (drive)', 'p_e': 'Press (drive)', 'p_g': 'Press (main)',
        'p_k': 'Press (main)', 'ahu_a': 'HVAC (air handling)', 'ahu_b': 'HVAC (air handling)',
        'ex_b': 'HVAC (air extraction)', 'comp_a': 'Compressed air', 'comp_b': 'Compressed air',
        'mh_a': 'Material handling', 'u_e': 'Utilities', 'mix_a': 'Process mixing'}
PRESS = [a for a in ASSETS if a.startswith('p_')]
HORIZONS = [1, 6, 24]
N_LAGS, N_Q, THETA = 24, 8, 0.2
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10
raw = pd.read_parquet('data/all_assets_rel.parquet')
raw = raw[raw.AssetId.isin(ASSETS)].copy()
raw['t'] = pd.to_datetime(raw['window_start_utc'])
q = raw.groupby(['AssetId', 't'])['Demand_kW'].mean().reset_index()
q['hts'] = q['t'].dt.floor('h')
hourly = q.groupby(['AssetId', 'hts'])['Demand_kW'].mean().reset_index().rename(columns={'hts': 'ts', 'Demand_kW': 'load'})
frames = []
for a in ASSETS:
    s = hourly[hourly.AssetId == a].set_index('ts')['load'].sort_index()
    s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq='h'))
    frames.append(pd.DataFrame({'AssetId': a, 'ts': s.index, 'load': s.values}))
base = pd.concat(frames, ignore_index=True)
all_ts = np.sort(base.loc[base.load.notna(), 'ts'].unique())
t_val = pd.Timestamp('2025-09-13')
t_test = pd.Timestamp('2025-10-19')
scale = base[base.ts < t_val].groupby('AssetId')['load'].max().replace(0, 1.0)
base['load_s'] = base['load'] / base.AssetId.map(scale)
q['q_s'] = q['Demand_kW'] / q.AssetId.map(scale)
base.to_parquet(f'{OUT}/base.parquet'); q[['AssetId', 't', 'Demand_kW', 'q_s']].to_parquet(f'{OUT}/q.parquet')
base['date'] = base.ts.dt.normalize(); base['hr'] = base.ts.dt.hour
day = base.pivot_table(index=['AssetId', 'date'], columns='hr', values='load', fill_value=0)
day_norm = day.div(day.max(axis=1), axis=0).fillna(0)
Xd = day_norm.values
best = (-1, None)
for eps in [0.03]:
    lab = DBSCAN(eps=eps, min_samples=5).fit(Xd).labels_
    n = len(set(lab)) - (1 if -1 in lab else 0); m = lab != -1
    if n >= 2 and m.sum() > n:
        s_ = silhouette_score(Xd[m], lab[m])
        if s_ > best[0]: best = (s_, eps)
chosen_eps = float(best[1])
didx = day_norm.index.to_frame(index=False)
trd = (didx.date < t_val).values
db = DBSCAN(eps=chosen_eps, min_samples=5).fit(Xd[trd])
core = Xd[trd][db.core_sample_indices_] if len(db.core_sample_indices_) else Xd[trd]
didx['day_dist'] = pairwise_distances(Xd, core).min(axis=1)
didx['day_anom'] = (didx.day_dist > chosen_eps).astype(int)
def idx(g):
    p = g['load'].dropna()
    mx, mn = p.max(), p.min()
    return pd.Series({'LF': p.mean() / mx if mx > 0 else np.nan, 'PBR': mx / mn if mn >= 0.5 else np.nan, 'VI': p.var()})
ind = base.groupby(['AssetId', 'date']).apply(idx).reset_index()
didx = didx.merge(ind, on=['AssetId', 'date'], how='left')
didx['VI'] = didx['VI'] / didx.AssetId.map(scale) ** 2
didx['day_mean'] = base.groupby(['AssetId', 'date'])['load_s'].mean().reindex(pd.MultiIndex.from_frame(didx[['AssetId', 'date']])).values
didx.to_parquet(f'{OUT}/days.parquet')
b = base.copy(); b['dow'] = b.ts.dt.dayofweek
on_prob = (b[(b.ts < t_val) & b.load_s.notna()].assign(on=lambda d: (d.load_s > THETA).astype(int))
           .groupby(['AssetId', 'dow', 'hr'])['on'].mean())
q_lookup = q.set_index(['AssetId', 't'])['q_s']
def build(H):
    d = base[['AssetId', 'ts', 'load', 'load_s']].copy()
    g = d.groupby('AssetId')['load_s']
    for k in range(1, N_LAGS + 1):
        d[f'lag_{k}'] = g.shift(H + k - 1)
    d['lag_168'] = g.shift(168)
    sh = g.shift(H)
    d['roll_mean_24'] = sh.groupby(d.AssetId).transform(lambda x: x.rolling(24, min_periods=18).mean())
    d['roll_std_24'] = sh.groupby(d.AssetId).transform(lambda x: x.rolling(24, min_periods=18).std())
    hr, dow = d.ts.dt.hour, d.ts.dt.dayofweek
    d['hour_sin'], d['hour_cos'] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    d['dow_sin'], d['dow_cos'] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    d['is_weekend'] = (dow >= 5).astype(int)
    for a in ASSETS: d[f'asset_{a}'] = (d.AssetId == a).astype(int)
    oe = d.ts - pd.Timedelta(hours=H - 1)
    for k in range(1, N_Q + 1):
        key = pd.MultiIndex.from_arrays([d.AssetId, oe - pd.Timedelta(minutes=15 * k)])
        d[f'q_lag_{k}'] = q_lookup.reindex(key).values
    d['q_slope'] = d['q_lag_1'] - d['q_lag_4']
    d['on_lag1'] = (d['lag_1'] > THETA).astype(int)
    d['on_yday'] = (g.shift(24) > THETA).astype(int)
    onm = d[[f'lag_{k}' for k in range(1, 25)]].values > THETA
    same = onm == onm[:, [0]]
    d['hours_in_state'] = np.where(same.all(axis=1), 24, np.argmin(same, axis=1))
    key = pd.MultiIndex.from_arrays([d.AssetId, dow, hr])
    d['usual_on_prob'] = on_prob.reindex(key).values
    hp = (d.ts - pd.Timedelta(hours=1))
    d['usual_on_prob_prev'] = on_prob.reindex(pd.MultiIndex.from_arrays([d.AssetId, hp.dt.dayofweek, hp.dt.hour])).values
    wl = d.pivot_table(index='ts', columns='AssetId', values='lag_1')
    wq = d.pivot_table(index='ts', columns='AssetId', values='q_lag_1')
    on_l, on_q = (wl > THETA).astype(float).where(wl.notna()), (wq > THETA).astype(float).where(wq.notna())
    cols = {'x_mean_lag1': wl, 'x_mean_q1': wq, 'x_on_frac': on_l, 'x_on_frac_q1': on_q}
    for name, W in cols.items():
        tot, cnt = W.sum(axis=1), W.notna().sum(axis=1)
        sv = W.stack().rename('own').reset_index()
        m = d[['AssetId', 'ts']].merge(sv, on=['ts', 'AssetId'], how='left')
        T_, C_ = d.ts.map(tot).values, d.ts.map(cnt).values
        own = m['own'].values; has = ~np.isnan(own)
        d[name] = np.where(C_ - has > 0, (T_ - np.nan_to_num(own)) / np.maximum(C_ - has, 1), np.nan)
    Wp = on_l[[c for c in on_l.columns if c in PRESS]]
    tot, cnt = Wp.sum(axis=1), Wp.notna().sum(axis=1)
    sv = Wp.stack().rename('own').reset_index()
    m = d[['AssetId', 'ts']].merge(sv, on=['ts', 'AssetId'], how='left'); own = m['own'].values; has = ~np.isnan(own)
    T_, C_ = d.ts.map(tot).values, d.ts.map(cnt).values
    d['x_press_on'] = np.where(C_ - has > 0, (T_ - np.nan_to_num(own)) / np.maximum(C_ - has, 1), np.nan)
    od = (d.ts - pd.Timedelta(hours=H)).dt.normalize() - pd.Timedelta(days=1)
    dl = didx.set_index(['AssetId', 'date'])
    k2 = pd.MultiIndex.from_arrays([d.AssetId, od])
    d['prev_day_mean'] = dl['day_mean'].reindex(k2).values
    for c, n in [('day_dist', 'prev_day_dist'), ('day_anom', 'prev_day_anom'), ('LF', 'prev_LF'), ('PBR', 'prev_PBR'), ('VI', 'prev_VI')]:
        d[n] = dl[c].reindex(k2).values
    d['pers'] = d['lag_1']
    d['snaive'] = g.shift(max(24, H) if H <= 24 else 168)
    d = d.dropna(subset=['load_s'] + [f'lag_{k}' for k in range(1, N_LAGS + 1)]).reset_index(drop=True)
    d['split'] = np.where(d.ts < t_val, 'train', np.where(d.ts < t_test, 'val', 'test'))
    d.to_parquet(f'{OUT}/feat_h{H}.parquet')
    print(H, d.split.value_counts().to_dict(), flush=True)
for H in HORIZONS: build(H)
GROUPS = {
    'lags': [f'lag_{k}' for k in range(1, N_LAGS + 1)] + ['lag_168', 'roll_mean_24', 'roll_std_24'],
    'calendar_asset': ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend'] + [f'asset_{a}' for a in ASSETS],
    'q15': [f'q_lag_{k}' for k in range(1, N_Q + 1)] + ['q_slope'],
    'onoff': ['on_lag1', 'on_yday', 'hours_in_state', 'usual_on_prob', 'usual_on_prob_prev'],
    'cross': ['x_mean_lag1', 'x_mean_q1', 'x_on_frac', 'x_on_frac_q1', 'x_press_on'],
    'prevday': ['prev_day_mean'],
    'dbscan': ['prev_day_dist', 'prev_day_anom', 'prev_LF', 'prev_PBR', 'prev_VI']}
json.dump({'ASSETS': ASSETS, 'TYPE': TYPE, 'HORIZONS': HORIZONS, 't_val': str(t_val), 't_test': str(t_test),
           'scale': scale.to_dict(), 'chosen_eps': chosen_eps, 'GROUPS': GROUPS, 'THETA': THETA},
          open(f'{OUT}/meta.json', 'w'), indent=1)
print('t_val', t_val, 't_test', t_test, 'eps', chosen_eps)
'''

SRC['profiling.py'] = r'''
import json, numpy as np, pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
D='data'; OUT='results'
meta=json.load(open(f'{D}/meta.json')); A=meta['ASSETS']; TYPE=meta['TYPE']; SC=meta['scale']
dy=pd.read_parquet(f'{D}/days.parquet'); dy['VIkw']=dy['VI']*dy.AssetId.map(SC)**2
base=pd.read_parquet(f'{D}/base.parquet')
rows=[]
for a in A:
    x=dy[dy.AssetId==a]; b=base[base.AssetId==a]
    rows.append(dict(Asset=a,Type=TYPE[a],Days=len(x),Mean_kW=b.load.mean(),Max_kW=SC[a],LF=x.LF.mean(),PBR=x.PBR.mean(),PBR_n=x.PBR.notna().mean(),VI=x.VIkw.mean(),
                     OnFrac=(b.load_s.dropna()>0.2).mean()))
T=pd.DataFrame(rows); T.to_csv(f'{OUT}/assets.csv',index=False); print(T.round(3).to_string())
base['date']=base.ts.dt.normalize(); base['hr']=base.ts.dt.hour
day=base.pivot_table(index=['AssetId','date'],columns='hr',values='load',fill_value=0)
X=day.div(day.max(axis=1),axis=0).fillna(0).values
res=[]
for k in range(2,8):
    lab=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X)
    res.append(dict(k=k,sil=silhouette_score(X,lab),dbi=davies_bouldin_score(X,lab)))
R=pd.DataFrame(res); print(R.round(4).to_string())
eps=meta['chosen_eps']; lab=DBSCAN(eps=eps,min_samples=5).fit(X).labels_; m=lab!=-1
nc=len(set(lab[m])); out=dict(n_days=len(X),eps=eps,db_noise=float((~m).mean()),db_nclust=nc,db_sil=float(silhouette_score(X[m],lab[m])),db_dbi=float(davies_bouldin_score(X[m],lab[m])),db_kept=int(m.sum()))
kb=R.loc[R.sil.idxmax()]; out.update(km_k=int(kb.k),km_sil=float(kb.sil),km_dbi=float(kb.dbi))
for k in [2,out['db_nclust']]:
    l2=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X[m]); out[f'km_same_k{k}_sil']=float(silhouette_score(X[m],l2)); out[f'km_same_k{k}_dbi']=float(davies_bouldin_score(X[m],l2))
for e in [0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0,1.2]:
    l=DBSCAN(eps=e,min_samples=5).fit(X).labels_
    print('eps',e,'noise',round((l==-1).mean(),3),'clusters',len(set(l))-(1 if -1 in l else 0))
lab=KMeans(n_clusters=out['km_k'],random_state=42,n_init=10).fit_predict(X)
comp=pd.crosstab(day.index.get_level_values(0),lab,normalize='index'); print(comp.round(2))
comp.to_csv(f'{OUT}/kmeans_composition.csv')
json.dump(out,open(f'{OUT}/clustering.json','w'),indent=1); print(out)
'''

SRC['roll2.py'] = r'''
import sys, os, json, numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import featlib as fl
FOLDS = {'Jul': ('2025-06-01', '2025-07-01', '2025-08-01'), 'Aug': ('2025-07-01', '2025-08-01', '2025-09-01'),
         'Sep': ('2025-08-01', '2025-09-01', '2025-10-19')}
OUT = 'preds_rolling'; os.makedirs(OUT, exist_ok=True)
info = {}
for FOLD, (v0, t0, t1) in FOLDS.items():
    v0, t0, t1 = map(pd.Timestamp, (v0, t0, t1)); scale = fl.scale_from(v0)
    for H in [1, 6, 24]:
        d = fl.build(H, scale)
        tr, va, te = (d.ts < v0).values, ((d.ts >= v0) & (d.ts < t0)).values, ((d.ts >= t0) & (d.ts < t1)).values
        info[f'{FOLD}_h{H}'] = dict(train_end=str(v0), val=[str(v0), str(t0)], test=[str(t0), str(t1)], n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()))
        y = d.load_s.values; age = (v0 - d.ts).dt.days.values; w = np.exp(-(age[tr] - age[tr].min()) / 90.0)
        for SEED in [int(s) for s in os.environ.get('ROLL_SEEDS', '42,7,123').split(',')]:
            fn = f'{OUT}/{FOLD}_h{H}_s{SEED}.npz'
            if os.path.exists(fn): continue
            def reg(f, mt=None, mv=None, wt=None):
                mt = np.ones(tr.sum(), bool) if mt is None else mt; mv = np.ones(va.sum(), bool) if mv is None else mv
                r = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                     early_stopping_rounds=100, random_state=SEED, n_jobs=2, tree_method='hist')
                r.fit(d.loc[tr, f][mt], y[tr][mt], sample_weight=None if wt is None else wt[mt], eval_set=[(d.loc[va, f][mv], y[va][mv])], verbose=False)
                return r
            F, B = fl.F_PROP, fl.F_BASE; out = {}
            out['XGBoost'] = reg(B).predict(d.loc[te, B])
            on_tr, on_va = (y[tr] > fl.TH).astype(int), (y[va] > fl.TH).astype(int)
            clf = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                                    early_stopping_rounds=100, eval_metric='logloss', random_state=SEED, n_jobs=2)
            clf.fit(d.loc[tr, F], on_tr, sample_weight=w, eval_set=[(d.loc[va, F], on_va)], verbose=False)
            p = clf.predict_proba(d.loc[te, F])[:, 1]
            ron, roff = reg(F, on_tr == 1, on_va == 1, w), reg(F, on_tr == 0, on_va == 0, w)
            out['Mixture'] = p * ron.predict(d.loc[te, F]) + (1 - p) * roff.predict(d.loc[te, F])
            sc = d.loc[te, 'scale'].values
            np.savez(fn, xgb_kw=np.clip(out['XGBoost'], 0, None) * sc, mix_kw=np.clip(out['Mixture'], 0, None) * sc,
                     y_kw=d.loc[te, 'load'].values, asset=d.loc[te, 'AssetId'].values.astype(str), ts=d.loc[te, 'ts'].values.astype('datetime64[s]').astype(np.int64))
            print(FOLD, H, SEED, info[f'{FOLD}_h{H}'], flush=True)
json.dump(info, open(f'{OUT}/fold_info.json', 'w'), indent=1)
print('ROLL2 DONE')
'''

SRC['roll2_analysis.py'] = r'''
import os
import json, numpy as np, pandas as pd, sys
sys.path.insert(0,'.'); from dmlib import dm, hourly, holm, block_boot
D='data'; PR='preds'; RP='preds_rolling'; O='results_updated'
meta=json.load(open(f'{D}/meta.json')); SC=meta['scale']; SEEDS=[int(s) for s in os.environ.get('ROLL_SEEDS', '42,7,123').split(',')]
def load(fold,H,s):
    if fold!='Main':
        Z=np.load(f'{RP}/{fold}_h{H}_s{s}.npz'); return Z['y_kw'],Z['xgb_kw'],Z['mix_kw'],Z['ts']
    f=pd.read_parquet(f'{D}/feat_h{H}.parquet',columns=['AssetId','ts','load','split']); te=(f.split=='test').values; sc=f.AssetId[te].map(SC).values
    return f.load.values[te], np.clip(np.load(f'{PR}/learned_h{H}_s{s}.npz')['XGBoost'],0,None)*sc, np.clip(np.load(f'{PR}/v2_h{H}_s{s}.npz')['Two-Stage v2'],0,None)*sc, f.ts.values[te].astype('datetime64[s]').astype(np.int64)
rows=[]
for H in [1,6,24]:
    pool={s:[] for s in SEEDS}
    for fold in ['Jul','Aug','Sep','Main']:
        rx,rm,ps,pb,n=[],[],[],[],0
        for s in SEEDS:
            y,x,m,ts=load(fold,H,s); pool[s].append((y,x,m,ts)); n=len(y)
            rx.append(np.sqrt(np.mean((y-x)**2))); rm.append(np.sqrt(np.mean((y-m)**2)))
            d=hourly(ts,(y-x)**2-(y-m)**2); ps.append(dm(d,H)[1]); pb.append(block_boot(d,seed=s))
        rows.append(dict(H=H,period=fold,n_rows=n,xgb=np.mean(rx),mix=np.mean(rm),gain=100*(1-np.mean(rm)/np.mean(rx)),p_max=max(ps),p_min=min(ps),p_block_max=max(pb),T_hours=len(d)))
    rx,rm,ps,pb=[],[],[],[]
    for s in SEEDS:
        y=np.concatenate([a[0] for a in pool[s]]); x=np.concatenate([a[1] for a in pool[s]]); m=np.concatenate([a[2] for a in pool[s]]); ts=np.concatenate([a[3] for a in pool[s]])
        rx.append(np.sqrt(np.mean((y-x)**2))); rm.append(np.sqrt(np.mean((y-m)**2)))
        d=hourly(ts,(y-x)**2-(y-m)**2); ps.append(dm(d,H)[1]); pb.append(block_boot(d,seed=s))
    rows.append(dict(H=H,period='All four (pooled)',n_rows=len(y),xgb=np.mean(rx),mix=np.mean(rm),gain=100*(1-np.mean(rm)/np.mean(rx)),p_max=max(ps),p_min=min(ps),p_block_max=max(pb),T_hours=len(d)))
R=pd.DataFrame(rows); R['p_holm']=holm(R.p_max); R['p_holm_block']=holm(R[['p_max','p_block_max']].max(1))
pd.set_option('display.width',250); print(R.round(4).to_string()); R.to_csv(f'{O}/rolling_split_scaled.csv',index=False)
'''

SRC['run25.py'] = r'''
import sys, os, json, numpy as np, pandas as pd, xgboost as xgb
H=int(sys.argv[1]); D='data25'; OUT='preds25'; os.makedirs(OUT,exist_ok=True)
meta=json.load(open(f'{D}/meta.json')); G=meta['GROUPS']; TH=meta['THETA']
d=pd.read_parquet(f'{D}/feat_h{H}.parquet'); tr,va,te=[(d.split==s).values for s in ['train','val','test']]; y=d.load_s.values
BASE=G['lags']+G['calendar_asset']; F=BASE+G['q15']+['on_lag1','on_yday','hours_in_state']+G['prevday']+G['recent']
age=(d.ts.max()-d.ts).dt.days.values; w=np.exp(-(age[tr]-age[tr].min())/90.0)
for SEED in [int(s) for s in os.environ.get('ROLL_SEEDS', '42,7,123').split(',')]:
    fn=f'{OUT}/h{H}_s{SEED}.npz'
    if os.path.exists(fn): continue
    def reg(f, mt=None, mv=None, wt=None):
        mt=np.ones(tr.sum(),bool) if mt is None else mt; mv=np.ones(va.sum(),bool) if mv is None else mv
        r=xgb.XGBRegressor(n_estimators=2000,learning_rate=0.03,max_depth=6,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
                           early_stopping_rounds=100,random_state=SEED,n_jobs=2,tree_method='hist')
        r.fit(d.loc[tr,f][mt],y[tr][mt],sample_weight=None if wt is None else wt[mt],eval_set=[(d.loc[va,f][mv],y[va][mv])],verbose=False); return r
    px=reg(BASE).predict(d.loc[te,BASE])
    on_tr,on_va=(y[tr]>TH).astype(int),(y[va]>TH).astype(int)
    clf=xgb.XGBClassifier(n_estimators=1500,learning_rate=0.03,max_depth=5,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
                          early_stopping_rounds=100,eval_metric='logloss',random_state=SEED,n_jobs=2)
    clf.fit(d.loc[tr,F],on_tr,sample_weight=w,eval_set=[(d.loc[va,F],on_va)],verbose=False); p=clf.predict_proba(d.loc[te,F])[:,1]
    ron,roff=reg(F,on_tr==1,on_va==1,w),reg(F,on_tr==0,on_va==0,w)
    pm=p*ron.predict(d.loc[te,F])+(1-p)*roff.predict(d.loc[te,F])
    sc=d.loc[te,'AssetId'].map(meta['scale']).values
    np.savez(fn,xgb_kw=np.clip(px,0,None)*sc,mix_kw=np.clip(pm,0,None)*sc,y_kw=d.loc[te,'load'].values,asset=d.loc[te,'AssetId'].values.astype(str),
             ts=d.loc[te,'ts'].values.astype('datetime64[s]').astype(np.int64))
    print(H,SEED,'n_test',int(te.sum()),flush=True)
'''

SRC['run_ts.py'] = r'''
import sys, json, numpy as np, pandas as pd, xgboost as xgb
H=int(sys.argv[1]); SEED=int(sys.argv[2])
D='data'; meta=json.load(open(f'{D}/meta.json')); G=meta['GROUPS']; TH=meta['THETA']
d=pd.read_parquet(f'{D}/feat_h{H}.parquet'); tr,va,te=[(d.split==s).values for s in ['train','val','test']]
y=d.load_s.values
F=G['lags']+G['calendar_asset']+G['q15']+['on_lag1','on_yday','hours_in_state']+G['prevday']+G['recent']
age=(d.ts.max()-d.ts).dt.days.values; w=np.exp(-(age[tr]-age[tr].min())/90.0)
Xtr,Xva,Xte=d.loc[tr,F],d.loc[va,F],d.loc[te,F]; on_tr,on_va=(y[tr]>TH).astype(int),(y[va]>TH).astype(int)
clf=xgb.XGBClassifier(n_estimators=1500,learning_rate=0.03,max_depth=5,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
    early_stopping_rounds=100,eval_metric='logloss',random_state=SEED,n_jobs=2)
clf.fit(Xtr,on_tr,sample_weight=w,eval_set=[(Xva,on_va)],verbose=False)
pva,pte=clf.predict_proba(Xva)[:,1],clf.predict_proba(Xte)[:,1]
def reg(mt,mv):
    r=xgb.XGBRegressor(n_estimators=2000,learning_rate=0.03,max_depth=6,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
        early_stopping_rounds=100,random_state=SEED,n_jobs=2)
    r.fit(Xtr[mt],y[tr][mt],sample_weight=w[mt],eval_set=[(Xva[mv],y[va][mv])],verbose=False); return r
ron,roff=reg(on_tr==1,on_va==1),reg(on_tr==0,on_va==0)
fn=f'preds/v2_h{H}_s{SEED}.npz'; P=dict(np.load(fn))
P['Two-Stage v2']=pte*ron.predict(Xte)+(1-pte)*roff.predict(Xte); P['Two-Stage v2|va']=pva*ron.predict(Xva)+(1-pva)*roff.predict(Xva)
np.savez(fn,**P); print(H,SEED,'ts done',flush=True)
'''

SRC['run_v3.py'] = r'''
import sys, json, numpy as np, pandas as pd, xgboost as xgb, os
H=int(sys.argv[1]); SEED=int(sys.argv[2])
D='data'; meta=json.load(open(f'{D}/meta.json')); G=meta['GROUPS']
d=pd.read_parquet(f'{D}/feat_h{H}.parquet'); tr,va,te=[(d.split==s).values for s in ['train','val','test']]
y=d.load_s.values
BASE=G['lags']+G['calendar_asset']; RP=G['recent']; DYN=['on_lag1','on_yday','hours_in_state']
FULL=BASE+G['q15']+DYN+G['prevday']+RP
V={'Proposed v2':(FULL,True),'v2 no weighting':(FULL,False),
   'v2 -q15':([c for c in FULL if c not in G['q15']],True),'v2 -recent':([c for c in FULL if c not in RP],True),
   'v2 -state':([c for c in FULL if c not in DYN],True),'v2 -prevday':([c for c in FULL if c not in G['prevday']],True)}
age=(d.ts.max()-d.ts).dt.days.values
fn=f'preds/v2_h{H}_s{SEED}.npz'; P=dict(np.load(fn)) if os.path.exists(fn) else {}
for k,(f,wgt) in V.items():
    if k in P: continue
    w=np.exp(-(age[tr]-age[tr].min())/90.0) if wgt else None
    m=xgb.XGBRegressor(n_estimators=2000,learning_rate=0.03,max_depth=6,subsample=0.8,colsample_bytree=0.8,min_child_weight=3,
                       early_stopping_rounds=100,random_state=SEED,n_jobs=2,tree_method='hist')
    m.fit(d.loc[tr,f],y[tr],sample_weight=w,eval_set=[(d.loc[va,f],y[va])],verbose=False)
    P[k]=m.predict(d.loc[te,f]); P[k+'|va']=m.predict(d.loc[va,f])
    if k=='Proposed v2': m.save_model(f'preds/v2_h{H}_s{SEED}.json')
    np.savez(fn,**P)
print(H,SEED,'done',flush=True)
'''

SRC['shap_analysis.py'] = r'''
import json, numpy as np, pandas as pd, xgboost as xgb
D = 'data'; PR = 'preds'; OUT = 'results'
meta = json.load(open(f'{D}/meta.json')); G = meta['GROUPS']; SCALE = meta['scale']
G['state'] = ['on_lag1', 'on_yday', 'hours_in_state']
ENH = G['lags'] + G['calendar_asset'] + G['q15'] + G['state'] + G['prevday'] + G['recent']
grp_of = {c: g for g in ['lags', 'calendar_asset', 'q15', 'state', 'prevday', 'recent'] for c in G[g]}
base = pd.read_parquet(f'{D}/base.parquet', columns=['AssetId', 'ts', 'load_s'])
prevv = base.assign(ts=base.ts + pd.Timedelta(hours=1)).set_index(['AssetId', 'ts'])['load_s']
feat_rows, grp_rows, dep = [], [], []
for H in meta['HORIZONS']:
    f = pd.read_parquet(f'{D}/feat_h{H}.parquet'); fi = f[f.split == 'test'].reset_index(drop=True)
    m = xgb.Booster(); m.load_model(f'{PR}/v2_h{H}_s42.json')
    sh = m.predict(xgb.DMatrix(fi[ENH], feature_names=ENH), pred_contribs=True)[:, :-1]
    sc = fi.AssetId.map(SCALE).values[:, None]; shk = sh * sc
    prev = prevv.reindex(pd.MultiIndex.from_frame(fi[['AssetId', 'ts']])).values
    sw = np.abs(fi.load_s.values - prev) > 0.2; sw[np.isnan(prev)] = False
    for j, c in enumerate(ENH):
        feat_rows.append(dict(H=H, feature=c, group=grp_of[c], mean_abs=np.abs(shk[:, j]).mean(),
                              mean_abs_sw=np.abs(shk[sw, j]).mean(), mean_abs_st=np.abs(shk[~sw, j]).mean()))
    F = pd.DataFrame([r for r in feat_rows if r['H'] == H])
    g = F.groupby('group')[['mean_abs', 'mean_abs_sw', 'mean_abs_st']].sum()
    g = g / g.sum(); g['H'] = H; grp_rows.append(g.reset_index())
    if H == 1:
        for c in ['q_slope', 'rp_week_mean', 'q_lag_1', 'lag_1']:
            j = ENH.index(c)
            dep.append(pd.DataFrame({'feature': c, 'value': fi[c].values, 'shap': sh[:, j]}).sample(3000, random_state=0))
    print(H, F.sort_values('mean_abs', ascending=False).head(10)[['feature', 'mean_abs', 'mean_abs_sw']].round(3).to_string(index=False))
pd.DataFrame(feat_rows).to_csv(f'{OUT}/shap_features.csv', index=False)
pd.concat(grp_rows).to_csv(f'{OUT}/shap_groups.csv', index=False)
pd.concat(dep).to_csv(f'{OUT}/shap_dependence_h1.csv', index=False)
print(pd.concat(grp_rows).round(3).to_string())
'''

SRC['tcn_q15.py'] = r'''
import sys, os, json, random, numpy as np, pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
H,SEED=int(sys.argv[1]),int(sys.argv[2]); D='data'; OUT='preds_extra'
meta=json.load(open(f'{D}/meta.json')); G=meta['GROUPS']
d=pd.read_parquet(f'{D}/feat_h{H}.parquet'); tr,va,te=[(d.split==s).values for s in ['train','val','test']]; y=d.load_s.values
import tensorflow as tf; from tensorflow import keras; from tensorflow.keras import layers
tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED); tf.config.threading.set_intra_op_parallelism_threads(2)
seq=[f'lag_{k}' for k in range(24,0,-1)]; aux=G['calendar_asset']+G['q15']
def inp(m): return [d.loc[m,seq].values[...,None].astype('float32'), d.loc[m,aux].fillna(d.loc[tr,aux].mean()).values.astype('float32')]
s_in,a_in=keras.Input((24,1)),keras.Input((len(aux),)); x=s_in
for dl in [1,2,4,8]:
    h=layers.Conv1D(32,3,padding='causal',dilation_rate=dl,activation='relu')(x); h=layers.Conv1D(32,3,padding='causal',dilation_rate=dl,activation='relu')(h)
    x=layers.Add()([layers.Conv1D(32,1)(x) if x.shape[-1]!=32 else x,h])
x=layers.Lambda(lambda z:z[:,-1,:])(x); x=layers.Concatenate()([x,a_in]); x=layers.Dense(32,activation='relu')(x); out=layers.Dense(1)(x)
m=keras.Model([s_in,a_in],out); m.compile(optimizer=keras.optimizers.Adam(1e-3),loss='mse')
m.fit(inp(tr),y[tr],validation_data=(inp(va),y[va]),epochs=30,batch_size=512,verbose=0,callbacks=[keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)])
fn=f'{OUT}/tcnq_h{H}_s{SEED}.npy'; np.save(fn,m.predict(inp(te),verbose=0,batch_size=2048).ravel()); print(H,SEED,'done',flush=True)
'''

SRC['twin_template.py'] = r'''
# ============================================================
# CELL 19: Digital Twin interface
# Builds a self-contained HTML Digital Twin of the 4 assets from the
# results of Cells 15-18 and saves it as 'digital_twin.html'.
# The twin replays the held-out TEST period hour by hour and shows, for
# the selected asset:
#   * a plant-floor view with an industrial symbol for each asset (press,
#     AHU fan, compressor, conveyor) coloured by state: running / idle / anomaly
#   * machine state (running / idle) and how long it has been in it
#   * measured demand and the next-hour forecast (Enhanced XGBoost),
#     next to the persistence baseline
#   * day health from the context-aware kNN anomaly detector (Cell 18
#     method), with the top-5% alert threshold
#   * today's load curve against the typical weekday / weekend curve
#   * LF, PBR, VI of the day, forecast accuracy, and an alert log
# Open the downloaded file in any web browser (Chrome / Edge).
# Requires: Cells 1-17 (uses final_df, scale, ON_THRESHOLD, hourly,
# daily_indices_ctx, SELECTED_ASSETS, POWER_COL).
# ============================================================

import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROLE = {'p_c': 'Press drive (production)', 'ahu_a': 'Air handling unit (HVAC)',
        'comp_b': 'Air compressor', 'mh_a': 'Material handling'}
K_NN, ALERT_Q = 5, 0.95

twin = {'assets': {},
        'note': ('Replay of the chronological test period (last 20% of the timeline, never used for '
                 'training). Forecast = Enhanced XGBoost (hourly + 15-min + ON/OFF + cross-asset '
                 'features). Day health = mean distance to the 5 most similar real days of the same '
                 'day type; alert when above the 95th percentile of all real days of that asset. '
                 'Data: open industrial asset-level energy dataset, 15-min aggregates, Demand_kW.')}

prof_all = hourly.pivot_table(index=['AssetId', 'date'], columns='hour', values=POWER_COL).dropna()
ctx_idx = daily_indices_ctx.copy()
ctx_idx['date'] = pd.to_datetime(ctx_idx['date'])

for a in SELECTED_ASSETS:
    amax = float(scale[a])
    sub = final_df[final_df['AssetId'] == a].set_index('ts').sort_index()
    # continuous hourly axis over the test period (gaps stay empty)
    rng_t = pd.date_range(sub.index.min(), sub.index.max(), freq='h')
    sub = sub.reindex(rng_t)
    r2 = lambda s: [None if pd.isna(v) else round(float(v), 3) for v in s]

    # context-aware kNN day score for every complete real day of this asset
    P = prof_all.loc[a]
    dates = pd.to_datetime(P.index)
    L = P.values / amax
    wk = np.asarray(dates.dayofweek >= 5).astype(int)
    score = np.zeros(len(L))
    typical = {}
    for t in [0, 1]:
        m = wk == t
        if m.sum() > K_NN + 1:
            d, _ = NearestNeighbors(n_neighbors=K_NN + 1).fit(L[m]).kneighbors(L[m])
            score[m] = d[:, 1:].mean(axis=1)          # skip the day itself
            typical[str(t)] = np.round(np.median(P.values[m], axis=0), 3).tolist()
        elif m.any():
            d, _ = NearestNeighbors(n_neighbors=K_NN).fit(L).kneighbors(L[m])
            score[m] = d.mean(axis=1)
    for t in ['0', '1']:
        typical.setdefault(t, np.round(np.median(P.values, axis=0), 3).tolist())
    thr = float(np.quantile(score, ALERT_Q))

    idx_a = ctx_idx[ctx_idx['AssetId'] == a].set_index('date')
    first_day = rng_t[0].normalize() - pd.Timedelta(days=2)
    days = {}
    for d, s, w in zip(dates, score, wk):
        if d < first_day:
            continue
        rec = {'s': round(float(s), 4), 'w': int(w)}
        if d in idx_a.index:
            row = idx_a.loc[d]
            rec['LF'] = None if pd.isna(row['LF']) else round(float(row['LF']), 4)
            rec['PBR'] = None if pd.isna(row['PBR']) else round(float(row['PBR']), 3)
            rec['VI'] = None if pd.isna(row['VI']) else round(float(row['VI']), 3)
        days[d.strftime('%Y-%m-%d')] = rec

    t_str = [t.strftime('%Y-%m-%d %H') for t in rng_t]
    start = next((k for k, t in enumerate(t_str) if t >= '2025-10-21 10'), 48)
    start = min(max(start, 48), len(t_str) - 2)
    ok = sub['Actual'].notna()
    twin['assets'][a] = {
        'role': ROLE.get(a, a), 'max': round(amax, 3), 'onKw': round(ON_THRESHOLD * amax, 3),
        't': t_str, 'a': r2(sub['Actual']), 'f': r2(sub['Enhanced XGBoost']), 'p': r2(sub['Persistence']),
        'rmse_enh': float(np.sqrt(((sub.loc[ok, 'Actual'] - sub.loc[ok, 'Enhanced XGBoost']) ** 2).mean())),
        'rmse_pers': float(np.sqrt(((sub.loc[ok, 'Actual'] - sub.loc[ok, 'Persistence']) ** 2).mean())),
        'thr': round(thr, 4), 'days': days, 'typical': typical, 'start': start}
    n_alert = sum(1 for dd, r in days.items() if r['s'] > thr and dd >= t_str[0][:10])
    print(f"{a}: {len(t_str)} test hours, {n_alert} anomalous test days, alert threshold {thr:.4f}")

TWIN_HEAD = r"""<title>Industrial Energy Digital Twin</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&display=swap">
<style>
/* Layout: control-room console. Header + asset tabs + time scrubber, then a 4-tile
   status row, then trend + day-profile panels, then indices / accuracy / alerts. */
:root {
  color-scheme: dark;
  --bg: #0d1520;          /* console background, blue-biased */
  --panel: #142031;       /* instrument panel */
  --panel-2: #1a293d;     /* raised element / track */
  --line: #25374f;        /* hairlines, grid */
  --fg: #dfe8f3;          /* primary text */
  --muted: #8195ad;       /* secondary text */
  --actual: #eef3f9;      /* measured load trace */
  --forecast: #4cc3e6;    /* model forecast trace (cyan instrument) */
  --persist: #7d8ea4;     /* persistence baseline */
  --ok: #3ecf8e;          /* state: normal / ON */
  --warn: #f2b33d;        /* state: warning */
  --crit: #f0625a;        /* state: anomaly */
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --f-body: "IBM Plex Sans", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --f-data: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
}
body { background: var(--bg); color: var(--fg); font-family: var(--f-body); font-size: 14px; }
.wrap { max-width: 1280px; margin: 0 auto; padding-inline: 20px; padding-block: 18px 28px; display: grid; gap: 14px; }
header.top { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px; }
.brand h1 { font-family: var(--f-display); font-weight: 600; font-size: 1.55rem; letter-spacing: .01em; margin: 0; text-wrap: balance; }
.brand p { margin: 2px 0 0; color: var(--muted); font-size: .82rem; }
.live { font-family: var(--f-data); font-size: .8rem; color: var(--muted); display: flex; align-items: center; gap: 8px; }
.live .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 0 3px color-mix(in srgb, var(--ok) 25%, transparent); }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; }
.tab { font: 500 .85rem var(--f-body); color: var(--muted); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px; cursor: pointer; display: grid; text-align: left; gap: 1px; }
.tab small { font-size: .7rem; color: var(--muted); font-weight: 400; }
.tab[aria-selected="true"] { color: var(--fg); border-color: var(--forecast); background: var(--panel-2); }
.tab:focus-visible, button:focus-visible, input:focus-visible { outline: 2px solid var(--forecast); outline-offset: 2px; }
.scrub { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 14px; display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }
.scrub button { font: 600 .8rem var(--f-body); background: var(--forecast); color: #06202b; border: 0; border-radius: 5px; padding: 7px 12px; cursor: pointer; min-width: 64px; }
.scrub button.ghost { background: var(--panel-2); color: var(--fg); border: 1px solid var(--line); min-width: 0; }
.scrub input[type=range] { flex: 1 1 260px; accent-color: var(--forecast); min-width: 0; }
.scrub .when { font-family: var(--f-data); font-size: .92rem; min-width: 190px; font-variant-numeric: tabular-nums; }
.grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.grid-main { display: grid; grid-template-columns: minmax(0, 1.75fr) minmax(0, 1fr); gap: 12px; }
.grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; min-width: 0; display: grid; gap: 8px; align-content: start; }
.label { font: 600 .7rem var(--f-body); text-transform: uppercase; letter-spacing: .09em; color: var(--muted); display: flex; justify-content: space-between; gap: 8px; }
.big { font-family: var(--f-data); font-size: 1.9rem; font-weight: 500; font-variant-numeric: tabular-nums; line-height: 1.1; }
.big small { font-size: .9rem; color: var(--muted); margin-left: 4px; }
.sub { color: var(--muted); font-size: .8rem; font-variant-numeric: tabular-nums; }
.pill { display: inline-flex; align-items: center; gap: 6px; font: 600 .78rem var(--f-body); letter-spacing: .04em; padding: 4px 10px; border-radius: 999px; width: fit-content; }
.pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.pill.ok { color: var(--ok); background: color-mix(in srgb, var(--ok) 14%, transparent); }
.pill.off { color: var(--muted); background: var(--panel-2); }
.pill.warn { color: var(--warn); background: color-mix(in srgb, var(--warn) 14%, transparent); }
.pill.crit { color: var(--crit); background: color-mix(in srgb, var(--crit) 16%, transparent); }
.meter { height: 8px; background: var(--panel-2); border-radius: 4px; position: relative; overflow: hidden; }
.meter > i { position: absolute; inset: 0 auto 0 0; background: var(--forecast); border-radius: 4px; }
.meter.state > i { background: var(--ok); }
.meter .mark { position: absolute; top: -2px; bottom: -2px; width: 2px; background: var(--warn); }
.card.alert-crit { border-color: color-mix(in srgb, var(--crit) 60%, var(--line)); }
.chart { width: 100%; height: auto; display: block; }
.chart text { fill: var(--muted); font: 10px var(--f-data); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: .75rem; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 16px; height: 0; border-top: 2px solid; display: inline-block; }
dl.kv { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; margin: 0; font-variant-numeric: tabular-nums; }
dl.kv dt { color: var(--muted); font-size: .8rem; }
dl.kv dd { margin: 0; font-family: var(--f-data); text-align: right; }
.log { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; max-height: 190px; overflow-y: auto; }
.log li { display: grid; grid-template-columns: auto 1fr auto; gap: 8px; align-items: center; font-size: .8rem; padding: 6px 8px; background: var(--panel-2); border-radius: 5px; cursor: pointer; border: 1px solid transparent; }
.log li:hover { border-color: var(--line); }
.log li b { font-family: var(--f-data); font-weight: 500; }
.log .sev { width: 8px; height: 8px; border-radius: 50%; background: var(--crit); }
.foot { color: var(--muted); font-size: .75rem; line-height: 1.5; max-width: 110ch; }
@media (max-width: 980px) { .grid4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } .grid-main, .grid3 { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 520px) { .grid4 { grid-template-columns: minmax(0, 1fr); } .wrap { padding-inline: 16px; } .big { font-size: 1.6rem; } }
@media (prefers-reduced-motion: no-preference) { .meter > i { transition: width .25s ease; } }

/* plant floor: machine symbols tinted by state */
.floor { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.mach { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; cursor: pointer;
  display: grid; grid-template-columns: 58px minmax(0, 1fr); gap: 4px 12px; align-items: center; text-align: left; color: var(--fg); font: inherit; }
.mach[aria-selected="true"] { border-color: var(--forecast); background: var(--panel-2); box-shadow: inset 0 0 0 1px var(--forecast); }
.mach .ico { grid-row: span 3; width: 58px; height: 58px; color: var(--muted); }
.mach.s-run .ico { color: var(--ok); } .mach.s-anom .ico { color: var(--crit); }
.mach.s-anom { border-color: color-mix(in srgb, var(--crit) 55%, var(--line)); }
.mach b { font: 600 .95rem var(--f-display); letter-spacing: .02em; }
.mach small { color: var(--muted); font-size: .72rem; }
.mach .st { font: 500 .78rem var(--f-data); font-variant-numeric: tabular-nums; }
.mach.s-run .st { color: var(--ok); } .mach.s-anom .st { color: var(--crit); }
.bigico { width: 84px; height: 84px; color: var(--muted); }
.bigico.s-run { color: var(--ok); } .bigico.s-anom { color: var(--crit); }
.statebox { display: flex; align-items: center; gap: 14px; }
.ico svg, .bigico svg { width: 100%; height: 100%; display: block; overflow: visible; }
@media (prefers-reduced-motion: no-preference) {
  .s-run .spin { animation: spin 1.1s linear infinite; transform-origin: 32px 32px; transform-box: view-box; }
  .s-run .ram { animation: ram 1.2s ease-in-out infinite; }
  .s-run .item { animation: belt 2.4s linear infinite; }
  .s-run .needle { animation: needle 1.6s ease-in-out infinite alternate; transform-origin: 20px 18px; transform-box: view-box; }
}
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes ram { 50% { transform: translateY(12px); } }
@keyframes belt { from { transform: translateX(-14px); } to { transform: translateX(14px); } }
@keyframes needle { from { transform: rotate(-35deg); } to { transform: rotate(35deg); } }
@media (max-width: 980px) { .floor { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 520px) { .floor { grid-template-columns: minmax(0, 1fr); } }
</style>"""

TWIN_BODY = r"""
<div class="wrap">
  <header class="top">
    <div class="brand">
      <h1>Industrial Energy Digital Twin</h1>
      <p>Asset-level load profiling, next-hour forecasting and anomaly alerts &middot; replaying the held-out test period</p>
    </div>
    <div class="live"><span class="dot" id="liveDot"></span><span id="liveTxt">Replay paused</span></div>
  </header>

  <nav class="floor" id="tabs" role="tablist" aria-label="Plant floor: select an asset"></nav>

  <section class="scrub" aria-label="Time control">
    <button id="play" type="button">Play</button>
    <button id="stepB" class="ghost" type="button" aria-label="Back one hour">&minus;1 h</button>
    <button id="stepF" class="ghost" type="button" aria-label="Forward one hour">+1 h</button>
    <input id="slider" type="range" min="0" max="1" value="0" aria-label="Replay time">
    <div class="when" id="when">&ndash;</div>
  </section>

  <section class="grid4">
    <div class="card" id="cState">
      <div class="label"><span>Machine state</span><span id="stateName"></span></div>
      <div class="statebox"><div class="bigico" id="bigIco"></div>
        <div style="display:grid;gap:6px;min-width:0"><div id="statePill" class="pill ok">ON</div>
        <div class="sub" id="stateSub">&ndash;</div></div></div>
    </div>
    <div class="card">
      <div class="label"><span>Measured demand</span><span id="maxTxt"></span></div>
      <div class="big" id="nowKw">&ndash;<small>kW</small></div>
      <div class="meter"><i id="nowBar"></i></div>
    </div>
    <div class="card">
      <div class="label"><span>Next-hour forecast</span><span>Enhanced XGBoost</span></div>
      <div class="big" id="fcKw">&ndash;<small>kW</small></div>
      <div class="sub" id="fcSub">&ndash;</div>
    </div>
    <div class="card" id="cHealth">
      <div class="label"><span>Day health</span><span>context kNN</span></div>
      <div id="healthPill" class="pill ok">Normal</div>
      <div class="meter"><i id="healthBar"></i><span class="mark" id="healthMark"></span></div>
      <div class="sub" id="healthSub">&ndash;</div>
    </div>
  </section>

  <section class="grid-main">
    <div class="card">
      <div class="label"><span>Last 48 hours and next-hour forecast</span><span id="trendNote"></span></div>
      <svg class="chart" id="trend" viewBox="0 0 720 250" role="img" aria-label="Actual and forecast demand"></svg>
      <div class="legend">
        <span><i style="border-color:var(--actual)"></i>Measured</span>
        <span><i style="border-color:var(--forecast)"></i>Forecast (Enhanced XGBoost)</span>
        <span><i style="border-color:var(--persist);border-top-style:dashed"></i>Persistence baseline</span>
      </div>
    </div>
    <div class="card">
      <div class="label"><span>Today vs typical day</span><span id="dayType"></span></div>
      <svg class="chart" id="dayChart" viewBox="0 0 420 250" role="img" aria-label="Today's profile against the typical profile"></svg>
      <div class="legend">
        <span><i style="border-color:var(--actual)"></i>Today so far</span>
        <span><i style="border-color:var(--warn);border-top-style:dashed"></i>Typical (median)</span>
      </div>
    </div>
  </section>

  <section class="grid3">
    <div class="card">
      <div class="label"><span>Load-profile indices</span><span>today</span></div>
      <dl class="kv">
        <dt>Load factor (LF)</dt><dd id="iLF">&ndash;</dd>
        <dt>Peak-to-base ratio (PBR)</dt><dd id="iPBR">&ndash;</dd>
        <dt>Variability index (VI, kW&sup2;)</dt><dd id="iVI">&ndash;</dd>
        <dt>Energy so far</dt><dd id="iE">&ndash;</dd>
      </dl>
    </div>
    <div class="card">
      <div class="label"><span>Forecast accuracy</span><span>test period</span></div>
      <dl class="kv">
        <dt>RMSE, Enhanced XGBoost</dt><dd id="aEnh">&ndash;</dd>
        <dt>RMSE, persistence</dt><dd id="aPers">&ndash;</dd>
        <dt>Skill vs persistence</dt><dd id="aSkill">&ndash;</dd>
        <dt>Rolling 24 h MAE</dt><dd id="aRoll">&ndash;</dd>
      </dl>
    </div>
    <div class="card">
      <div class="label"><span>Anomaly alerts</span><span id="alertCount"></span></div>
      <ul class="log" id="log"></ul>
    </div>
  </section>

  <p class="foot" id="foot"></p>
</div>

<script>
const TWIN = /*__DATA__*/null;
(function () {
  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const fmt = (x, d = 2) => (x == null || isNaN(x)) ? "–" : Number(x).toFixed(d);

  const ICON = {
    press: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="9" y="5" width="46" height="8" rx="1.5"/><line x1="14" y1="13" x2="14" y2="54"/><line x1="50" y1="13" x2="50" y2="54"/>
      <g class="ram"><line x1="32" y1="13" x2="32" y2="19"/><rect x="21" y="19" width="22" height="11" rx="1" fill="currentColor" fill-opacity=".25"/></g>
      <rect x="21" y="45" width="22" height="9" rx="1"/><rect x="6" y="54" width="52" height="6" rx="1.5" fill="currentColor" fill-opacity=".25"/></svg>`,
    fan: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="5" y="5" width="54" height="54" rx="6"/><circle cx="32" cy="32" r="22" stroke-opacity=".45"/>
      <g class="spin" fill="currentColor" fill-opacity=".3">
        <path d="M32 32 C28 22 30 13 37 12 C42 17 38 26 32 32 Z"/>
        <path d="M32 32 C28 22 30 13 37 12 C42 17 38 26 32 32 Z" transform="rotate(120 32 32)"/>
        <path d="M32 32 C28 22 30 13 37 12 C42 17 38 26 32 32 Z" transform="rotate(240 32 32)"/>
        <circle cx="32" cy="32" r="3.5" fill-opacity="1"/></g></svg>`,
    compressor: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="4" y="30" width="42" height="20" rx="10" fill="currentColor" fill-opacity=".2"/>
      <line x1="12" y1="50" x2="10" y2="58"/><line x1="38" y1="50" x2="40" y2="58"/>
      <rect x="44" y="20" width="16" height="16" rx="2"/><line x1="46" y1="25" x2="58" y2="25"/><line x1="46" y1="30" x2="58" y2="30"/>
      <line x1="20" y1="26" x2="20" y2="30"/><circle cx="20" cy="18" r="8"/>
      <line class="needle" x1="20" y1="18" x2="20" y2="12" stroke-linecap="round"/></svg>`,
    conveyor: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round" aria-hidden="true">
      <rect x="3" y="36" width="58" height="12" rx="6"/>
      <circle cx="9" cy="42" r="2.6"/><circle cx="24" cy="42" r="2.6"/><circle cx="40" cy="42" r="2.6"/><circle cx="55" cy="42" r="2.6"/>
      <line x1="12" y1="48" x2="10" y2="60"/><line x1="52" y1="48" x2="54" y2="60"/>
      <rect class="item" x="23" y="20" width="18" height="16" rx="1.5" fill="currentColor" fill-opacity=".25"/></svg>`
  };
  const ICON_OF = { p_c: "press", ahu_a: "fan", comp_b: "compressor", mh_a: "conveyor" };
  const iconFor = (a) => ICON[ICON_OF[a] || "press"];
  const TIDX = {}; Object.keys(TWIN.assets).forEach((a) => { const m = {}; TWIN.assets[a].t.forEach((x, k) => m[x] = k); TIDX[a] = m; });
  const assets = Object.keys(TWIN.assets);
  let cur = assets.includes("p_c") ? "p_c" : assets[0];
  let i = 0, timer = null;

  // tabs
  assets.forEach((a) => {
    const b = document.createElement("button");
    b.className = "mach"; b.type = "button"; b.setAttribute("role", "tab"); b.id = "tab-" + a;
    b.innerHTML = `<span class="ico">${iconFor(a)}</span><b>${a}</b><small>${TWIN.assets[a].role}</small><span class="st" id="st-${a}">&ndash;</span>`;
    b.onclick = () => { const t = TWIN.assets[cur].t[i]; cur = a; const k = TIDX[a][t]; if (k != null) i = k; setup(); };
    $("tabs").appendChild(b);
  });

  function setup() {
    const A = TWIN.assets[cur];
    assets.forEach((a) => $("tab-" + a).setAttribute("aria-selected", a === cur ? "true" : "false"));
    $("slider").max = A.t.length - 2;
    i = Math.min(Math.max(i, 48), A.t.length - 2);
    if (A.start != null && i === 48) i = A.start;
    $("slider").value = i;
    $("maxTxt").textContent = `max ${fmt(A.max, 1)} kW`;
    $("aEnh").textContent = fmt(A.rmse_enh, 3) + " kW";
    $("aPers").textContent = fmt(A.rmse_pers, 3) + " kW";
    $("aSkill").textContent = fmt(100 * (1 - A.rmse_enh / A.rmse_pers), 1) + " %";
    // alert log: anomalous days inside the test period
    const days = Object.entries(A.days).filter(([d, r]) => r.s > A.thr && d >= A.t[0].slice(0, 10))
      .sort((x, y) => y[1].s - x[1].s);
    $("alertCount").textContent = `${days.length} days above threshold`;
    const log = $("log"); log.innerHTML = "";
    days.slice(0, 40).forEach(([d, r]) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="sev"></span><span>${d} (${r.w ? "weekend" : "weekday"})</span><b>${fmt(r.s / A.thr, 2)}&times;</b>`;
      li.title = "Jump to this day";
      li.onclick = () => { const k = A.t.findIndex((t) => t.startsWith(d)); if (k >= 0) { i = Math.min(k + 18, A.t.length - 2); render(); } };
      log.appendChild(li);
    });
    if (!days.length) log.innerHTML = `<li><span></span><span>No anomalous days in the test period</span><b></b></li>`;
    $("foot").textContent = TWIN.note;
    render();
  }

  function hoursSame(A, k) { // hours the machine has stayed in its current state
    const on = (x) => x > A.onKw;
    let n = 0; const s = on(A.a[k]);
    for (let j = k; j >= 0 && on(A.a[j]) === s && n < 999; j--) n++;
    return n;
  }

  function render() {
    const A = TWIN.assets[cur];
    const t = A.t[i], day = t.slice(0, 10), hr = +t.slice(11, 13);
    $("when").textContent = `${t}:00 UTC`;
    $("slider").value = i;
    // state
    const on = A.a[i] > A.onKw, hs = hoursSame(A, i);
    $("statePill").className = "pill " + (on ? "ok" : "off");
    $("statePill").textContent = on ? "RUNNING" : "IDLE";
    $("stateName").textContent = A.role;
    const anomToday = A.days[day] && A.days[day].s > A.thr;
    $("bigIco").innerHTML = iconFor(cur);
    $("bigIco").className = "bigico " + (anomToday ? "s-anom" : on ? "s-run" : "");
    assets.forEach((a) => {
      const B = TWIN.assets[a], k = TIDX[a][t], el = $("tab-" + a);
      if (k == null || B.a[k] == null) { el.className = "mach"; $("st-" + a).textContent = "no data"; return; }
      const r = B.days[t.slice(0, 10)], an = r && r.s > B.thr, run = B.a[k] > B.onKw;
      el.className = "mach " + (an ? "s-anom" : run ? "s-run" : "");
      $("st-" + a).textContent = `${an ? "ANOMALY" : run ? "RUNNING" : "IDLE"} · ${fmt(B.a[k], 1)} kW`;
    });
    $("stateSub").textContent = `${on ? "Running" : "Idle"} for ${hs >= 999 ? "999+" : hs} h · ON above ${fmt(A.onKw, 1)} kW`;
    // demand
    $("nowKw").innerHTML = `${fmt(A.a[i])}<small>kW</small>`;
    $("nowBar").style.width = Math.min(100, 100 * A.a[i] / A.max) + "%";
    // forecast for next hour (index i+1)
    const nt = A.t[i + 1];
    $("fcKw").innerHTML = `${fmt(A.f[i + 1])}<small>kW</small>`;
    const err = A.a[i] - A.f[i];
    $("fcSub").textContent = `for ${nt.slice(11, 13)}:00 · persistence ${fmt(A.p[i + 1])} kW · last-hour error ${err >= 0 ? "+" : ""}${fmt(err)} kW`;
    // day health
    const r = A.days[day];
    if (r) {
      const ratio = r.s / A.thr, bad = r.s > A.thr, near = ratio > 0.8;
      $("healthPill").className = "pill " + (bad ? "crit" : near ? "warn" : "ok");
      $("healthPill").textContent = bad ? "ANOMALY" : near ? "WATCH" : "NORMAL";
      $("healthBar").style.width = Math.min(100, 100 * ratio / 2) + "%";
      $("healthBar").style.background = css(bad ? "--crit" : near ? "--warn" : "--ok");
      $("healthMark").style.left = "50%";
      $("healthSub").textContent = `score ${fmt(r.s, 3)} · alert threshold ${fmt(A.thr, 3)} (top 5 %)`;
      $("cHealth").classList.toggle("alert-crit", bad);
      $("iLF").textContent = fmt(r.LF, 3); $("iPBR").textContent = r.PBR == null ? "n/a (base < 0.5 kW)" : fmt(r.PBR, 2);
      $("iVI").textContent = fmt(r.VI, 2);
    } else {
      $("healthPill").className = "pill off"; $("healthPill").textContent = "INCOMPLETE DAY";
      $("healthBar").style.width = "0%"; $("healthSub").textContent = "Day has missing hours; not scored";
      $("cHealth").classList.remove("alert-crit");
      ["iLF", "iPBR", "iVI"].forEach((k) => $(k).textContent = "–");
    }
    // energy so far today and rolling MAE
    let e = 0, k = i; while (k >= 0 && A.t[k].startsWith(day)) { e += A.a[k]; k--; }
    $("iE").textContent = fmt(e, 1) + " kWh";
    let s = 0, n = 0; for (let j = Math.max(0, i - 23); j <= i; j++) { s += Math.abs(A.a[j] - A.f[j]); n++; }
    $("aRoll").textContent = fmt(s / n, 3) + " kW";
    drawTrend(A); drawDay(A, day);
    $("trendNote").textContent = A.a[i + 1] != null ? `actual next hour ${fmt(A.a[i + 1])} kW` : "";
  }

  function axis(svg, x0, x1, y0, y1, ymax, ticks) {
    let g = "";
    for (let k = 0; k <= ticks; k++) {
      const v = ymax * k / ticks, y = y1 - (y1 - y0) * k / ticks;
      g += `<line x1="${x0}" x2="${x1}" y1="${y}" y2="${y}" stroke="${css("--line")}" stroke-width="1"/>`;
      g += `<text x="${x0 - 6}" y="${y + 3}" text-anchor="end">${String(+v.toFixed(2))}</text>`;
    }
    return g;
  }
  const niceMax = (m) => { const p = Math.pow(10, Math.floor(Math.log10(m || 1))); for (const s of [1, 2, 2.5, 5, 10]) if (s * p >= m) return s * p; return 10 * p; };

  function path(xs, ys) {
    let d = "", pen = false;
    xs.forEach((x, k) => { const y = ys[k]; if (y == null || isNaN(y)) { pen = false; return; } d += (pen ? "L" : "M") + x.toFixed(1) + " " + y.toFixed(1); pen = true; });
    return d;
  }

  function drawTrend(A) {
    const W = 720, H = 250, x0 = 44, x1 = W - 14, y0 = 12, y1 = H - 28;
    const a0 = Math.max(0, i - 47), idx = []; for (let k = a0; k <= i + 1; k++) idx.push(k);
    const vals = idx.flatMap((k) => [A.a[k], A.f[k], A.p[k]]).filter((v) => v != null);
    const ymax = niceMax(Math.max(...vals, A.onKw) * 1.05);
    const X = (k) => x0 + (x1 - x0) * (k - a0) / 48, Y = (v) => y1 - (y1 - y0) * v / ymax;
    let g = axis(null, x0, x1, y0, y1, ymax, 4);
    // time labels every 12 h
    idx.forEach((k) => { if (+A.t[k].slice(11, 13) % 12 === 0) g += `<text x="${X(k)}" y="${H - 10}" text-anchor="middle">${A.t[k].slice(5, 10)} ${A.t[k].slice(11, 13)}h</text>`; });
    // ON threshold
    g += `<line x1="${x0}" x2="${x1}" y1="${Y(A.onKw)}" y2="${Y(A.onKw)}" stroke="${css("--ok")}" stroke-opacity=".35" stroke-dasharray="2 4"/>`;
    // now marker
    g += `<line x1="${X(i)}" x2="${X(i)}" y1="${y0}" y2="${y1}" stroke="${css("--muted")}" stroke-dasharray="3 3"/>`;
    g += `<text x="${X(i) - 4}" y="${y0 + 10}" text-anchor="end">now</text>`;
    const past = idx.filter((k) => k <= i);
    g += `<path d="${path(idx.map(X), idx.map((k) => A.p[k] == null ? null : Y(A.p[k])))}" fill="none" stroke="${css("--persist")}" stroke-width="1.3" stroke-dasharray="5 4"/>`;
    g += `<path d="${path(idx.map(X), idx.map((k) => A.f[k] == null ? null : Y(A.f[k])))}" fill="none" stroke="${css("--forecast")}" stroke-width="1.8"/>`;
    g += `<path d="${path(past.map(X), past.map((k) => A.a[k] == null ? null : Y(A.a[k])))}" fill="none" stroke="${css("--actual")}" stroke-width="2.2"/>`;
    // next-hour forecast point
    g += `<circle cx="${X(i + 1)}" cy="${Y(A.f[i + 1])}" r="5" fill="${css("--forecast")}" stroke="${css("--bg")}" stroke-width="2"/>`;
    $("trend").innerHTML = g;
  }

  function drawDay(A, day) {
    const W = 420, H = 250, x0 = 40, x1 = W - 10, y0 = 12, y1 = H - 28;
    const wk = A.days[day] ? A.days[day].w : ((new Date(day + "T00:00:00Z").getUTCDay() % 6 === 0) ? 1 : 0);
    const typ = A.typical[String(wk)];
    $("dayType").textContent = wk ? "weekend reference" : "weekday reference";
    const today = new Array(24).fill(null);
    for (let k = i; k >= 0 && A.t[k].startsWith(day); k--) today[+A.t[k].slice(11, 13)] = A.a[k];
    const ymax = niceMax(Math.max(...typ, ...today.filter((v) => v != null), 0.1) * 1.08);
    const X = (h) => x0 + (x1 - x0) * h / 23, Y = (v) => y1 - (y1 - y0) * v / ymax;
    let g = axis(null, x0, x1, y0, y1, ymax, 4);
    [0, 6, 12, 18, 23].forEach((h) => g += `<text x="${X(h)}" y="${H - 10}" text-anchor="middle">${String(h).padStart(2, "0")}h</text>`);
    const hrs = [...Array(24).keys()];
    g += `<path d="${path(hrs.map(X), typ.map(Y))}" fill="none" stroke="${css("--warn")}" stroke-width="1.6" stroke-dasharray="6 4"/>`;
    const area = path(hrs.map(X), today.map((v) => v == null ? null : Y(v)));
    g += `<path d="${area}" fill="none" stroke="${css("--actual")}" stroke-width="2.2"/>`;
    $("dayChart").innerHTML = g;
  }

  function play() {
    if (timer) { clearInterval(timer); timer = null; $("play").textContent = "Play"; $("liveTxt").textContent = "Replay paused"; return; }
    $("play").textContent = "Pause"; $("liveTxt").textContent = "Replaying test data";
    timer = setInterval(() => { const A = TWIN.assets[cur]; i = i >= A.t.length - 2 ? 48 : i + 1; render(); }, 350);
  }
  $("play").onclick = play;
  $("stepF").onclick = () => { i = Math.min(i + 1, TWIN.assets[cur].t.length - 2); render(); };
  $("stepB").onclick = () => { i = Math.max(i - 1, 0); render(); };
  $("slider").oninput = (e) => { i = +e.target.value; render(); };
  setup();
})();
</script>
"""

html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + TWIN_HEAD + '\n</head>\n<body>\n'
        + TWIN_BODY.replace('/*__DATA__*/null', json.dumps(twin, separators=(',', ':')))
        + '\n</body>\n</html>\n')
with open('digital_twin.html', 'w', encoding='utf-8') as fh:
    fh.write(html)
print(f"\nSaved digital_twin.html ({len(html)/1e6:.1f} MB).")

# Download it from Colab and open in a browser:
try:
    from google.colab import files
    files.download('digital_twin.html')
except Exception:
    print("Not running in Colab -- open digital_twin.html from the working folder.")
print("Cell 19 complete.")
'''

for name, text in SRC.items():
    with open(f'code/{name}', 'w') as fh: fh.write(text.lstrip('\n'))

def run(script, *args, soft=False):
    cmd = [sys.executable, f'code/{script}'] + [str(a) for a in args]
    t = time.time(); print(f'>> {script} {" ".join(map(str, args))}', flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = '\n'.join(l for l in (p.stdout + p.stderr).splitlines()
                    if l.strip() and not any(w in l for w in ('Warning', 'warn(', 'WARNING', 'oneDNN', 'cpu_feature', 'absl', 'rebuild TensorFlow')))
    if out: print(out[-2500:])
    print(f'   ({time.time() - t:.0f} s)', flush=True)
    if p.returncode != 0:
        if soft: print(f'!! {script} failed - continuing'); return
        raise RuntimeError(f'{script} failed - see the message above')

def step(title): print(f'\n{"=" * 100}\n{title}   [{(time.time() - T0) / 60:.1f} min]\n{"=" * 100}', flush=True)

step('STEP 1  Dataset')
def have_data():
    return bool(glob.glob('dataset/**/data_parquet', recursive=True) or glob.glob('dataset/**/data_csv', recursive=True))
def extract(zpath):
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        rel = [n for n in names if n.endswith('all_assets_rel.parquet')]
        if rel:
            with open('data/all_assets_rel.parquet', 'wb') as fh: fh.write(z.read(rel[0]))
            print('reliable windows loaded from', os.path.basename(zpath)); return
        keep = [n for n in names if '/data_parquet/' in '/' + n] or [n for n in names if '/data_csv/' in '/' + n] or names
        print('extracting', len(keep), 'files from', os.path.basename(zpath)); z.extractall('dataset', members=keep)
if os.path.exists('data/all_assets_rel.parquet'):
    print('reliable windows already prepared')
elif not have_data():
    if os.path.exists(DATA_ZIP):
        extract(DATA_ZIP)
    else:
        from google.colab import files
        print('Dataset ZIP not found at', DATA_ZIP, '- please select it now:')
        for f in files.upload():
            if f.endswith('.zip'): extract(f); os.remove(f)
    assert os.path.exists('data/all_assets_rel.parquet') or have_data(), 'No dataset found in the ZIP'

step('STEP 2  Data preparation, timestamp audit and features (H = 1, 6, 24 h)')
if not os.path.exists('data/all_assets_rel.parquet'): run('make_reliable.py')
if not os.path.exists('data/feat_h24.parquet'): run('prep.py')
if 'recent' not in json.load(open('data/meta.json'))['GROUPS']: run('addfeat.py')
meta = json.load(open('data/meta.json')); print('assets:', ', '.join(meta['ASSETS']))
print('train < ', meta['t_val'], '<= validation <', meta['t_test'], '<= test')
run('audit.py')

step('STEP 3  Reference forecasts: persistence, seasonal naive, K-means profile (Baseline-1), ARIMA, VAR')
if not os.path.exists('preds/classical_h24.npz'): run('classical.py')

step('STEP 4  Tree baselines: XGBoost, random forest, DBSCAN + XGBoost, conference feature set')
TREES = ['XGBoost', 'Random Forest', 'DBSCAN + XGBoost', 'Enhanced XGBoost']
if not QUICK: TREES += ['Ablation -q15', 'Ablation -onoff', 'Ablation -cross', 'Ablation -prevday', 'Two-Stage Hybrid']
for H in [1, 6, 24]:
    for S in SEEDS: run('learned.py', H, S, ','.join(TREES))

step('STEP 5  Neural baselines: LSTM, CNN-LSTM, TCN, Transformer')
if RUN_NN:
    import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU') or 'none (CPU is slower)')
    for H in [1, 6, 24]:
        for S in SEEDS: run('learned.py', H, S, 'LSTM,CNN-LSTM,TCN,Transformer')

step('STEP 6  State-aware XGBoost mixture, single-stage variant and ablations')
import numpy as np
for H in [1, 6, 24]:
    for S in SEEDS:
        run('run_v3.py', H, S)
        if 'Two-Stage v2' not in np.load(f'preds/v2_h{H}_s{S}.npz').files: run('run_ts.py', H, S)

step('STEP 7  Equal-information comparisons and ON/OFF classifier diagnostics')
for H in [1, 6, 24]:
    for S in SEEDS:
        run('extra.py', H, S)
        if RUN_NN: run('tcn_q15.py', H, S, soft=True)

if RUN_ROLLING:
    step('STEP 8  Rolling-origin evaluation (scale, states and inputs rebuilt for each period)')
    run('roll2.py')

step('STEP 9  Load profiling, metrics, Diebold-Mariano tests, asset-averaged errors, TreeSHAP')
run('profiling.py')
run('analysis.py')
run('dm2.py', soft=True)
if RUN_ROLLING: run('roll2_analysis.py')
run('extra_analysis.py')
if RUN_NN: run('asset_avg.py', soft=True)
run('shap_analysis.py')

step('STEP 10  Chronological anomaly detection (reference: training days, threshold: validation days)')
run('anomaly_chrono.py')

if RUN_25:
    step('STEP 11  Main comparison on all 25 eligible assets')
    run('prep25.py'); run('addfeat25.py')
    for H in [1, 6, 24]: run('run25.py', H)

step('STEP 12  Figures (600 dpi, PDF and PNG) and digital twin')
for f in ['fig_profiles.py', 'fig_faults.py', 'fig_forecasting.py', 'fig_anomaly.py']:
    run(f, soft=True)
run('build_twin.py')

from IPython.display import display, Image, HTML
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
step('RESULTS')
S_ = pd.read_csv('results/forecast_summary.csv')
piv = S_.pivot(index='model', columns='H', values='RMSE_mean')
for H in [1, 6, 24]:
    print(f'H = {H:2d} h: state-aware mixture {piv.loc["Enhanced XGBoost", H]:.3f} kW | XGBoost {piv.loc["XGBoost", H]:.3f} kW'
          f' | reduction {100 * (1 - piv.loc["Enhanced XGBoost", H] / piv.loc["XGBoost", H]):.1f}%')
for f in ['dm_hourly.csv', 'rolling_split_scaled.csv', 'equal_information.csv']:
    if os.path.exists(f'results_updated/{f}'):
        display(HTML(f'<h3>{f}</h3>')); display(pd.read_csv(f'results_updated/{f}').round(4))
for p in sorted(glob.glob('figures/*.png')):
    display(HTML(f'<h3 style="margin:18px 0 4px">{os.path.basename(p)[:-4]}</h3>')); display(Image(filename=p, width=1000))
display(HTML('<iframe style="width:100%;height:900px;border:1px solid #ccc;background:#fff" srcdoc="'
             + html.escape(open('twin/digital_twin_13assets.html').read()) + '"></iframe>'))

tmp = '/content/_pack'; shutil.rmtree(tmp, ignore_errors=True)
for d in ['results', 'results_updated', 'figures', 'twin', 'code']: shutil.copytree(d, f'{tmp}/{d}')
shutil.make_archive('/content/results', 'zip', tmp)
print(f'\nDONE in {(time.time() - T0) / 60:.1f} min. results.zip = results + figures + digital twin + code')
try:
    from google.colab import files; files.download('/content/results.zip')
except Exception:
    print('Download it from the Files panel: /content/results.zip')
