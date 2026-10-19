#!/usr/bin/env python3
# =====================================================================================================
#  export_predictions.py
#  Writes per-row (per-timestamp) forecast files for every asset from the artifacts that the main
#  analysis already produced. Run it in the same working folder that Updated_Python_Code.py used
#  (the folder that contains data/ and preds/), e.g. in Google Drive:
#      MyDrive/state_aware_forecasting/
#
#  It reads:
#     data/meta.json            scales, horizons, asset types
#     data/feat_h{H}.parquet    AssetId, ts, load_s, split  (test rows are used)
#     preds/v2_h{H}_s{seed}.npz key 'Two-Stage v2'  = proposed state-aware mixture (Enhanced XGBoost)
#     preds/classical_h{H}.npz  key 'Persistence'   = persistence baseline (optional reference column)
#
#  It writes, for each horizon H and seed:
#     predictions/forecast_h{H}_seed{seed}.csv
#        AssetId, timestamp_utc, horizon_h, seed, actual_kW, predicted_kW, persistence_kW
#  and a combined predictions/forecast_all_rows_seed42.csv for the manuscript's main seed.
#
#  It also prints per-asset test RMSE so you can check it against results/results/per_asset.csv
#  (rows with model == "Enhanced XGBoost", seed 42) before committing.
#
#  Dependencies: numpy, pandas  (already installed by the main script).
# =====================================================================================================

import os, json, glob
import numpy as np
import pandas as pd

ROOT = os.environ.get('ROOT', '.')          # folder holding data/ and preds/
D   = os.path.join(ROOT, 'data')
PR  = os.path.join(ROOT, 'preds')
OUT = os.path.join(ROOT, 'predictions')
os.makedirs(OUT, exist_ok=True)

meta  = json.load(open(os.path.join(D, 'meta.json')))
HS    = meta['HORIZONS']
SCALE = meta['scale']
TYPE  = meta.get('TYPE', {})
PROP_KEY = 'Two-Stage v2'                     # proposed model inside v2_h{H}_s{seed}.npz

def seeds_for(H):
    out = []
    for fn in sorted(glob.glob(os.path.join(PR, f'v2_h{H}_s*.npz'))):
        s = fn.split('_s')[-1].split('.npz')[0]
        if s.isdigit():
            out.append(int(s))
    return out

check_rows = []
main_seed_frames = []

for H in HS:
    f = pd.read_parquet(os.path.join(D, f'feat_h{H}.parquet'),
                        columns=['AssetId', 'ts', 'load_s', 'split'])
    te = (f.split == 'test').values
    fi = f[te].reset_index(drop=True)
    sc = fi.AssetId.map(SCALE).values
    actual = fi.load_s.values * sc            # observed demand, kW

    # optional persistence reference, aligned to the same test rows
    pers = None
    cfn = os.path.join(PR, f'classical_h{H}.npz')
    if os.path.exists(cfn):
        cl = np.load(cfn)
        if 'Persistence' in cl.files:
            pers = np.clip(np.nan_to_num(cl['Persistence'][te], nan=0.0), 0, None) * sc

    for seed in seeds_for(H):
        p = np.load(os.path.join(PR, f'v2_h{H}_s{seed}.npz'))[PROP_KEY]
        pred = np.clip(np.nan_to_num(p, nan=0.0), 0, None) * sc   # proposed forecast, kW

        df = pd.DataFrame({
            'AssetId': fi.AssetId.values,
            'timestamp_utc': pd.to_datetime(fi.ts.values),
            'horizon_h': H,
            'seed': seed,
            'actual_kW': actual,
            'predicted_kW': pred,
        })
        if pers is not None:
            df['persistence_kW'] = pers

        df.to_csv(os.path.join(OUT, f'forecast_h{H}_seed{seed}.csv'), index=False)

        if seed == 42:
            main_seed_frames.append(df)
            for a in fi.AssetId.unique():
                m = (fi.AssetId == a).values
                e = actual[m] - pred[m]
                check_rows.append(dict(H=H, Asset=a, Type=TYPE.get(a, ''),
                                       RMSE=float(np.sqrt(np.mean(e ** 2)))))

if main_seed_frames:
    pd.concat(main_seed_frames, ignore_index=True).to_csv(
        os.path.join(OUT, 'forecast_all_rows_seed42.csv'), index=False)

if check_rows:
    chk = pd.DataFrame(check_rows).sort_values(['H', 'Asset'])
    chk.to_csv(os.path.join(OUT, 'per_asset_rmse_check_seed42.csv'), index=False)
    print('Per-asset test RMSE (seed 42, proposed model) — compare with')
    print('results/results/per_asset.csv rows where model == "Enhanced XGBoost":')
    print(chk.round(3).to_string(index=False))

print('\nWrote per-row prediction files to:', OUT)
print('Files:', sorted(os.listdir(OUT)))
