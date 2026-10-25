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
        S['LOF (level vectors)']=-LocalOutlierFactor(n_neighbors=20,novelty=True).fit(lev[ref]).score_samples(lev)
        S['Isolation Forest (level vectors)']=-IsolationForest(n_estimators=300,random_state=sd).fit(lev[ref]).score_samples(lev)
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
Rr=pd.DataFrame(rows); Rr.to_csv(f'{O}/anomaly_extra_all.csv',index=False); AL=pd.concat(alerts)
pd.set_option('display.width',250)
print(Rr[(Rr.budget==0.05)|Rr.budget.isna()].groupby('Method')[['FA','All','AUC']+FAULTS].mean().round(1).sort_values('All').to_string())
print(Rr.groupby(['Method','budget']).All.mean().unstack().round(1).to_string())
print(Rr.groupby(['Method','budget']).FA.mean().unstack().round(1).to_string())
print(Rr[Rr.budget==0.05].groupby('Method')[['AUC','All_matched','FA_matched']+[k+' (matched)' for k in FAULTS]].mean().round(3).sort_values('All_matched').to_string())
print(Rr.groupby(['Method','budget']).All_matched.mean().unstack().round(1).to_string())
print('seed SD of All at 5%:', Rr[Rr.budget==0.05].groupby(['Method','seed']).All.mean().groupby('Method').std().round(2).to_dict())
print('alerts seed42 per asset:', AL.groupby('Asset').alert.sum().to_dict(), 'total', int(AL.alert.sum()))
