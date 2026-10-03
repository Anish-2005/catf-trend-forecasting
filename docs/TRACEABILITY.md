# Requirement traceability (RM -> design module -> prototype component)

| Req. | Requirement | Module | Prototype component | Status in this prototype |
|---|---|---|---|---|
| FR-01 | Ingest community-structured interaction data | M1 | `data.py` (synthetic + CSV) | Done (open-dataset connector pending) |
| FR-02 | Epoch binning and graph snapshots | M1, M2 | `graph.py` | Done |
| FR-03 | Community detection and tracking | M3 | `community.py` | Done (Louvain + overlap matching; GNN variant pending) |
| FR-04 | Activity and structural features | M4 | `features.py` | Done |
| FR-05 | Topic and sentiment features | M4 | `features.py`, `lexicon.py` | Basic (lexicon sentiment, NMF topics) |
| FR-06 | Activity forecasting | M5 | `forecast.py` | Done (naive, global / community ARIMA, hybrid) |
| FR-07 | Topic / sentiment forecasting, hybrid ARIMA-LSTM | M5 | `forecast.py` | Partial (MLP residual learner; LSTM and topic-probability forecast pending) |
| FR-08 | LT90 and benchmarking | M6 | `evaluate.py` | Basic (synthetic data only) |
| FR-09 | Early-warning alerts | M6 | `warning.py` | Basic (threshold rule) |
| FR-10 | Interactive dashboard | M7 | `app.py` | Basic (5 views) |
| NFR-03 | Reproducibility | all | fixed seeds, `config.yaml`, tests | Done |
| NFR-04 | Privacy | M1 | `data.anonymise()` | Helper provided |
