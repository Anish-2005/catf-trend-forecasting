# CATF Prototype - Community-Aware Temporal Forecasting of Trends in Social Media Networks

Group 07 · PCC-CSBS781 · 7th-semester prototype

End-to-end prototype of the CATF pipeline: interaction log → temporal graph snapshots → community
detection and lineage tracking → temporal + semantic features → forecasting (baselines and an
ARIMA + neural-residual hybrid) → LT90 early-warning evaluation → interactive dashboard.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py          # ~30-60 s, writes outputs/latest/
streamlit run app.py            # dashboard (results from outputs/latest are already included)
pytest -q                       # 11 tests
```

Options: `python run_pipeline.py --seed 7`, `--input my_data.csv`, `--config my_config.yaml`.

## Pipeline (maps to PRD modules)

| Module | File | What it does |
|---|---|---|
| M1 Ingestion | `catf/data.py` | Synthetic generator (planted, drifting communities and trend bursts) and CSV loader; pseudonymisation helper |
| M2 Temporal graph | `catf/graph.py` | Epoch binning; one weighted graph snapshot per epoch (3-epoch causal window) |
| M3 Communities | `catf/community.py` | Louvain per epoch; Jaccard matching gives persistent IDs and birth / continue / merge / split / death events |
| M4 Features | `catf/features.py` | Activity, active users, internal-interaction ratio, lexicon sentiment, NMF topics, topic novelty, growth |
| M5 Forecasting | `catf/forecast.py` | Naive, Global ARIMA, Community ARIMA, **Hybrid** (ARIMA + MLP on residuals and covariates); walk-forward, no look-ahead |
| M6 Evaluation / warning | `catf/evaluate.py`, `catf/warning.py` | MAE / RMSE / sMAPE, critical-mass events, alerts, event recall, precision, **LT90** |
| M7 Dashboard | `app.py` | Overview, Communities, Forecasts, Alerts, Model comparison; re-run or upload a CSV from the sidebar |

Outputs (`outputs/latest/`): `forecasts.csv`, `metrics_accuracy.csv`, `metrics_early_warning.csv`,
`alerts.csv`, `critical_events.csv`, `lineage_events.csv`, `membership.csv`, `features.csv`,
`panel.csv`, `topics.json`, `summary.json`.

## Using real data

Provide a CSV with columns `timestamp,user,target,text` (`target` = user replied to / mentioned;
see `data/sample_schema.csv`). Reddit / Pushshift-style comment dumps can be exported to this shape.
Then `python run_pipeline.py --input my_data.csv`. Use `catf.data.anonymise()` before sharing results.
Adjust `epoch.length_hours`, `community.min_size` and `forecast.min_coverage` in `config.yaml` to the dataset.

## Definitions used

* **Critical mass** of a community = mean + k·std of its activity in the training period (k = 2).
* **Alert** at epoch t: activity is still below critical mass, but the forecast crosses it within horizon f.
* **Lead time** of an event = event epoch − earliest alert epoch within [event − f, event − 1].
* **LT90** = lead time (epochs) reached or exceeded by 90 % of the *detected* events (10th percentile).
  This is a working definition: align it with the definition in the synopsis / PRD if they differ
  (`catf/evaluate.py::lt90`).

## Reference run (synthetic data, seed 42; 5 forecast communities, 9 test events, horizon 5)

| Model | MAE | Events detected | Alert precision | LT90 |
|---|---|---|---|---|
| Naive | 56.8 | 0 / 9 | - | - |
| Global ARIMA | 56.6 | 0 / 9 | - | - |
| Community ARIMA | 56.5 | 1 / 9 | 1.00 (2 alerts) | 2 |
| **Hybrid** | 56.7 | **4 / 9** | 0.45 (20 alerts) | 1 |

Community recovery: mean NMI 0.98 against planted communities. Across seeds 42 / 7 / 123 the hybrid
detected 44 % / 38 % / 86 % of events versus 11-14 % for Community ARIMA, with point accuracy (MAE)
about equal; alert precision was 0.45-0.59.

**Read this carefully.** The data is synthetic and the bursts carry *precursor* signals (emerging
vocabulary and sentiment shift) by construction, which is exactly what the hybrid's covariates can
see. These numbers show the pipeline works end to end and that covariates can add lead time; they are
not scientific evidence. Real-data benchmarking is the 8th-semester milestone.

## Known limitations (prototype scope)

* The neural residual learner is an sklearn MLP, **not yet the LSTM** (see `docs/ARCHITECTURE.md` for the plug-in point).
* Sentiment uses a small lexicon; topics use TF-IDF + NMF (fit on training text only).
* Communities come from Louvain on interaction graphs; GNN embeddings (GCN / GraphSAGE / GAT / TGN) are not included.
* Topic-emergence probability forecasting (FR-07) is approximated by topic-novelty covariates.
* Only communities present in ≥ 90 % of epochs are forecast; the split child and merged-away communities are tracked but not forecast.
* Hybrid point accuracy (MAE) is not better than ARIMA on this data; the gain is in early warning.

## Next steps (Sem 8)

PyTorch LSTM residual learner · GNN community embeddings · BERTopic / transformer sentiment · Prophet and
Per-Post LSTM baselines · real dataset benchmark · hyper-parameter search on a validation window · test-case suite for the dashboard.
