Dataset used in "State-Aware Multi-Horizon Load Forecasting and Daily Anomaly Detection for Industrial IoT Energy Monitoring"

Source
  C. Flynn, T. Murphy, J. Walsh, and D. Riordan. 2026. An open industrial energy dataset with asset-level
  measurements and high-coverage 15-minute aggregates from a manufacturing facility. Data 11, 5 (2026), 101.
  doi:10.3390/data11050101. License: CC BY 4.0. Data are redistributed here in reduced form under the same license.

File
  all_assets_rel.parquet - reliable three-phase 15-minute windows of all 43 assets
  (31 December 2024 to 31 December 2025, UTC; 1,016,182 rows)

Columns
  AssetId           asset identifier (e.g. p_b, ahu_a, comp_b)
  window_start_utc  start of the 15-minute window (UTC)
  Phase             ALL (three-phase total)
  IsReliableWindow  reliability flag of the dataset (only reliable windows are included)
  Demand_kW         average demand over the 15-minute window (kW)

Use with the code
  Put this ZIP in Google Drive as MyDrive/industrial_energy_dataset.zip and run Updated_Python_Code.py
  (or the .ipynb) in Google Colab. The code also accepts the original dataset ZIP.
