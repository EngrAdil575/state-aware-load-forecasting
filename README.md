# State-Aware Multi-Horizon Load Forecasting and Daily Anomaly Detection for Sub-Metered Industrial Assets

Code, data, trained offline digital twin, and results for the manuscript:

> **State-Aware Multi-Horizon Load Forecasting and Daily Anomaly Detection for Sub-Metered Industrial Assets**
> Adil Zohaib, Bilal Ahmed, Muhammad Amad Khan, Saqib Nawaz, Haris Sheh Zad, Sohail Khalid.

**Status:** This manuscript is currently submitted to ACM Transactions on Internet of Things (ACM TIoT) and is under review. It is not published. This repository is the supporting material for the submitted version.

This repository is the reproducibility record for the manuscript. It holds the dataset used in the study (in reduced form), the complete analysis code, the offline digital-twin interface, and every result file, figure, and supplementary table reported in the manuscript.

## What the manuscript does

The work presents an asset-level monitoring workflow for sub-metered industrial machines and evaluates it on an open year-long dataset of 13 industrial assets (25 in a supplementary check, 43 in the full inventory). The forecaster is a state-aware XGBoost mixture: a classifier estimates the probability that an asset is ON, and two regressors predict demand in the ON and OFF regimes from the latest 15-minute readings, the operating state, and recent load profiles recomputed at every forecast origin. Separate models are fit for 1, 6, and 24-hour horizons. Daily deviations are scored with a level-preserving nearest-neighbor rule that compares each day with the asset's earlier days of the same type.

## Repository layout

```
state-aware-load-forecasting/
├── data/                       Dataset (reduced form, CC BY 4.0 — see data/DATASET_README.txt)
│   ├── all_assets_rel.parquet  Reliable 15-min windows, 43 assets, 31 Dec 2024 – 31 Dec 2025 (1,016,182 rows)
│   └── DATASET_README.txt      Source, columns, and usage
├── code/                       Analysis code (Google Colab, single cell)
│   ├── Updated_Python_Code.py
│   ├── Updated_Python_Code.ipynb
│   └── requirements.txt
├── digital_twin/               Offline digital-twin replay interface
│   └── Updated_Digital_Twin.html
├── results/                    All results, figures, and supplementary tables
│   ├── results/                Core result tables (seed 42 and across seeds)
│   ├── results_updated/        Revised evaluation: rolling-origin, DM tests, robustness, anomaly scores
│   ├── figures/                Figures 1–11 as 600-dpi PDF and PNG
│   ├── Supplementary Material.xlsx
│   └── RESULTS_README.txt      File-by-file description, mapped to manuscript tables and figures
├── requirements.txt
├── CITATION.cff
└── LICENSE                     MIT (code). Data keep their original CC BY 4.0 license.
```

## Dataset

The dataset is redistributed in reduced form from:

> C. Flynn, T. Murphy, J. Walsh, and D. Riordan. 2026. *An open industrial energy dataset with asset-level measurements and high-coverage 15-minute aggregates from a manufacturing facility.* Data 11, 5 (2026), 101. doi:[10.3390/data11050101](https://doi.org/10.3390/data11050101). License: CC BY 4.0.

`data/all_assets_rel.parquet` contains the reliable three-phase 15-minute windows for all 43 assets. It is shared under the same CC BY 4.0 license as the original. The code also accepts the original dataset ZIP; see `data/DATASET_README.txt` and the header of the code for the exact input path.

## Reproducing the results

The analysis runs as a single cell in Google Colab.

1. Open `code/Updated_Python_Code.ipynb` in Google Colab, or paste `code/Updated_Python_Code.py` into one cell.
2. Set the runtime to a GPU (Runtime → Change runtime type → T4 GPU).
3. Place the dataset ZIP in Google Drive as `MyDrive/industrial_energy_dataset.zip`, or point `DATA_ZIP` at `data/all_assets_rel.parquet`.
4. Run the cell. It prepares the data, fits the forecaster and all baselines, runs the statistical tests, the rolling-origin and equal-information comparisons, the classifier diagnostics, the chronological anomaly detection, and TreeSHAP, then writes the result files, the 600-dpi figures, and the digital twin.

Flags at the top of the code control scope. `QUICK = True` runs one seed for a fast check; `QUICK = False` runs the five seeds used in the manuscript and takes several hours. `RUN_NN`, `RUN_ROLLING`, and `RUN_25` toggle the neural-network baselines, the rolling-origin evaluation, and the 25-asset repetition.

Dependencies are listed in `requirements.txt`. The main libraries are xgboost, lightgbm, scikit-learn, statsmodels, shap, pyarrow, pandas, numpy, scipy, matplotlib, and tensorflow (for the neural-network baselines).

## Offline digital twin

`digital_twin/Updated_Digital_Twin.html` is a self-contained offline replay interface (Figure 11 in the manuscript). Open it in any modern browser. It shows asset state, measured demand, the next-hour forecast of the state-aware mixture, the daily deviation score, and the last 48 hours against the typical weekday. It is an offline replay for illustration, with no live integration.

## Results

Every file in `results/` is produced by the code. `results/RESULTS_README.txt` describes each file and maps it to the tables and figures of the manuscript. Errors are in kW unless stated otherwise.

## Citation

If you use this code or data, please cite the manuscript (see `CITATION.cff`) and the original dataset of Flynn et al. (2026) above.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The dataset in `data/` keeps its original CC BY 4.0 license and is attributed to Flynn et al. (2026).
