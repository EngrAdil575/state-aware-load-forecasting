# Per-row (per-timestamp) predictions

This folder holds the per-timestamp test-set forecasts of the proposed state-aware
mixture (the "Enhanced XGBoost" model), one row per asset and 15-minute window.

## Files (once generated)

- `forecast_h{H}_seed{seed}.csv` — for each horizon H ∈ {1, 6, 24} and each seed,
  columns: `AssetId, timestamp_utc, horizon_h, seed, actual_kW, predicted_kW, persistence_kW`
- `forecast_all_rows_seed42.csv` — all horizons combined for the manuscript's main seed (42)
- `per_asset_rmse_check_seed42.csv` — per-asset test RMSE recomputed from these rows,
  for checking against `results/results/per_asset.csv` (rows with `model == "Enhanced XGBoost"`)

All values are in kW. `actual_kW` is the observed demand, `predicted_kW` is the proposed
model's forecast, `persistence_kW` is the persistence baseline for reference.

## How to generate

These files are produced from the artifacts that the main analysis already writes, so
they match the reported results exactly. In the same working folder that
`code/Updated_Python_Code.py` used (the one containing `data/` and `preds/` — on Google
Drive this is `MyDrive/state_aware_forecasting/`), run:

```bash
python code/export_predictions.py
```

or, in a Colab cell:

```python
%cd /content/drive/MyDrive/state_aware_forecasting
!python /path/to/export_predictions.py   # or paste the script into a cell
```

The script reads `data/feat_h{H}.parquet` (test rows) and `preds/v2_h{H}_s{seed}.npz`
(the proposed model's predictions), joins them, and writes the CSVs above. It takes a
few seconds because it reuses the already-trained outputs; it does not retrain anything.
