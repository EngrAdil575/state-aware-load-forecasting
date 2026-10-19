Results of "State-Aware Multi-Horizon Load Forecasting and Daily Anomaly Detection for Sub-Metered Industrial Assets"
All results are produced by Updated_Python_Code.py. Errors are in kW unless stated otherwise.

results/
  forecast_summary.csv      Tables 6, 7 and 9 (ablation rows): RMSE, MAE, MAPE, R2, skill, switching/steady RMSE, mean and SD over seeds
  forecast_all.csv          the same metrics for every seed
  per_asset.csv             test RMSE per asset (seed 42)
  assets.csv                Table 1 (profile indices per asset)
  clustering.json           Table 5 (clustering quality)
  kmeans_composition.csv    composition of the K-means clusters
  shap_groups.csv           Fig. 8 (TreeSHAP share of each feature group)
  shap_features.csv         Fig. 9 (mean absolute TreeSHAP attribution per feature)
  shap_dependence_h1.csv    TreeSHAP values of the 1-h model (seed 42)
  test_info.json            test rows, share of switching hours and persistence RMSE per horizon
  Model names: "Enhanced XGBoost" = state-aware mixture (proposed); "Single-stage enhanced" = single-stage variant;
  "Enhanced (conference features)" = conference feature set; "Ablation -x" = single-stage variant without group x.

results_updated/
  row_counts_per_asset_split_horizon.csv   rows per asset, split and horizon
  audit.json                               timestamp and row-assignment checks, switching hours per horizon
  asset_inventory_43.csv                   all 43 assets with the reason for inclusion or exclusion
  baseline_selection_val.json              validation RMSE used to select the strongest baseline
  dm_hourly.csv                            Diebold-Mariano tests on the hourly cross-asset loss differential (Holm-adjusted)
  rolling_split_scaled.csv                 Table 8 (rolling-origin evaluation with period-specific scaling)
  equal_information.csv                    Table 9 (equal-information rows)
  classifier_diagnostics.csv               Brier score, log loss, calibration error and negative outputs of the ON/OFF classifier
  classifier_reliability.csv               reliability of the ON probability
  asset_averaged_errors.csv                mean per-asset RMSE (kW and scaled target)
  per_asset_change_5seed.csv               Fig. 7 (per-asset RMSE change, five-seed mean)
  anomaly_chrono_all.csv                   Tables 10 and 11, Fig. 10 (chronological anomaly detection, all seeds and assets)
  anomaly_chrono_alerts_seed42.csv         daily deviation scores and alerts in the test period
  robustness_25_assets*.csv                main comparison on all 25 eligible assets

figures/   Figs. 1-11 of the manuscript as 600-dpi PDF and PNG
  fig_framework (1), fig_faults (2), fig_profiles (3), fig_horizon (4), fig_switching (5), fig_forecast (6),
  fig_assets (7), fig_shap_groups (8), fig_shap_top (9), fig_anomaly (10), fig_twin (11)

Supplementary Material.xlsx   supplementary tables of the manuscript
