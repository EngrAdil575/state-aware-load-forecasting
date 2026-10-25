Additional analyses added during pre-submission review (run after prep.py and addfeat.py, from the working folder):
  code/reviewer_extra_cost_bootstrap.py       Table 12 (training time, trees, model size, inference time) and the
                                              weekly block-bootstrap intervals of Section 6.3 (seed 42) -> refine_seed42.json
  code/reviewer_extra_anomaly_level_vectors.py  Table 10 rows "LOF, level vectors" and "Isolation Forest, level vectors"
                                              (same protocol as anomaly_chrono.py, five seeds) -> anomaly_extra_all.csv
  code/reviewer_extra_switching_error_share.py  Section 4.3: share of switching hours and of the squared error of XGBoost
                                              (base features, seed 42) on validation and test data -> switching_error_share.json
