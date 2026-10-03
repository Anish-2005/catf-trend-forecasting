# Architecture

```
interactions.csv ──► M1 ingest ──► M2 epoch + snapshot graphs ──► M3 Louvain + lineage tracking
                                                                        │ membership (epoch,user,cid)
                         M4 features (activity, sentiment, topics, structure) ◄──┘
                                   │ panel[cid][epoch]
                                   ▼
        M5 forecasting: naive | global ARIMA | community ARIMA | hybrid (ARIMA + residual learner)
                                   │ forecasts (model, cid, origin, h, y_true, y_pred)
                                   ▼
        M6 critical-mass events · alerts · MAE/RMSE/sMAPE · recall · LT90  ──►  outputs/latest/*.csv|json
                                   ▼
                              M7 Streamlit dashboard (app.py)
```

## Design decisions

* **Strictly chronological evaluation.** Parameters and thresholds are fitted on the training period; at
  each test origin the ARIMA state is *appended* with observed epochs (no refit, no look-ahead).
  The topic model is fitted on training-period text only.
* **Causal snapshots.** Snapshot G_t uses epochs t-2..t; the first two epochs are skipped (warm-up).
* **Log scale.** All models forecast `log1p(activity)` and are back-transformed.
* **Hybrid.** `forecast = ARIMA(t+h) + g(lagged residuals, covariates)`; covariates: sentiment delta,
  topic novelty, internal-interaction ratio, growth, log active users. One pooled residual model across
  communities, multi-output over horizons 1..f.

## Plugging in the LSTM (Sem 8)

`forecast._make_residual_model(kind, seed)` returns any object with `fit(X, Y)` / `predict(X)` where
X = `[residual lags..., covariates...]` and Y = residuals for horizons 1..f. Implement a PyTorch
sequence model with the same interface (or change `_row` to emit windows) and select it via
`forecast.residual_model` in `config.yaml`.
