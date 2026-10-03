"""M7 - CATF dashboard.   Run:  streamlit run app.py"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from catf.config import ROOT, load_config
from catf.pipeline import run

st.set_page_config(page_title="CATF Dashboard", page_icon="📈", layout="wide")
st.title("CATF - Community-Aware Temporal Forecasting")
st.caption("Prototype dashboard: communities, forecasts, early-warning alerts and model comparison")


@st.cache_data(show_spinner=False)
def load(out_dir: str, stamp: float):
    d = Path(out_dir)
    rd = lambda n: pd.read_csv(d / n)
    return dict(
        summary=json.loads((d / "summary.json").read_text()),
        topics=json.loads((d / "topics.json").read_text()),
        panel=rd("panel.csv"), forecasts=rd("forecasts.csv"), alerts=rd("alerts.csv"),
        events=rd("critical_events.csv"), lineage=rd("lineage_events.csv"), thr=rd("thresholds.csv"),
        acc=rd("metrics_accuracy.csv"), ew=rd("metrics_early_warning.csv"),
    )


# ---------------- sidebar: data + (re)run ----------------
cfg = load_config()
st.sidebar.header("Run")
out_dir = Path(st.sidebar.text_input("Output folder", str(ROOT / cfg["output_dir"])))
with st.sidebar.form("run"):
    seed = st.number_input("Seed", 0, 10_000, 42)
    epochs = st.number_input("Synthetic epochs (days)", 60, 300, 120)
    horizon = st.number_input("Forecast horizon (epochs)", 2, 10, 5)
    up = st.file_uploader("...or upload CSV (timestamp,user,target,text)", type="csv")
    go_run = st.form_submit_button("Run pipeline")
if go_run:
    over = {"data": {"synthetic": {"seed": int(seed), "n_epochs": int(epochs)}},
            "forecast": {"seed": int(seed), "horizon": int(horizon)}, "community": {"seed": int(seed)}}
    path = None
    if up is not None:
        tmp = Path(tempfile.mkdtemp()) / "upload.csv"
        tmp.write_bytes(up.getvalue())
        path = str(tmp)
    with st.spinner("Running pipeline (about 30-60 s)..."):
        try:
            run(load_config(overrides=over), path, str(out_dir), verbose=False)
            st.cache_data.clear()
            st.sidebar.success("Done")
        except Exception as exc:  # surface problems in the UI
            st.sidebar.error(str(exc))

if not (out_dir / "summary.json").exists():
    st.info("No results yet. Press **Run pipeline** in the sidebar (or `python run_pipeline.py`).")
    st.stop()

D = load(str(out_dir), (out_dir / "summary.json").stat().st_mtime)
S, panel = D["summary"], D["panel"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Communities", "Forecasts", "Alerts", "Model comparison"])

with tab1:
    c = st.columns(5)
    c[0].metric("Interactions", f"{S['n_interactions']:,}")
    c[1].metric("Epochs", S["n_epochs"])
    c[2].metric("Tracked communities", S["n_tracked_communities"])
    c[3].metric("Critical-mass events (test)", S["n_events"])
    c[4].metric("Community NMI vs planted", S.get("community_nmi_mean", "n/a"))
    if S.get("synthetic"):
        st.warning("Synthetic data: results validate the pipeline, they are not scientific evidence.")
    st.markdown("**Early-warning summary (test period)**")
    st.dataframe(D["ew"].round(2), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Community activity over time")
    fig = px.line(panel, x="epoch", y="activity", color=panel["cid"].astype(str), labels={"color": "community"})
    fig.add_vline(x=S["split_epoch"], line_dash="dot", annotation_text="train | test")
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    a.subheader("Sentiment per community")
    a.plotly_chart(px.line(panel, x="epoch", y="sentiment", color=panel["cid"].astype(str)), use_container_width=True)
    b.subheader("Topic novelty (precursor signal)")
    b.plotly_chart(px.line(panel, x="epoch", y="topic_novelty", color=panel["cid"].astype(str)), use_container_width=True)
    st.subheader("Community lineage (birth / merge / split / death)")
    lin = D["lineage"]
    st.dataframe(lin[lin["event"] != "continue"], use_container_width=True, hide_index=True)
    with st.expander("Discovered topics (top words)"):
        for k, w in D["topics"].items():
            st.write(f"**topic {k}**: {', '.join(w)}")

with tab3:
    f = D["forecasts"]
    cids = sorted(f["cid"].unique())
    cc = st.columns(3)
    cid = cc[0].selectbox("Community", cids)
    models = list(f["model"].unique())
    shown = cc[1].multiselect("Models", models, default=["community_arima", "hybrid"])
    h = cc[2].slider("Horizon h (epochs ahead)", 1, int(S["horizon"]), 1)
    base = panel[panel["cid"] == cid]
    fig = go.Figure(go.Scatter(x=base["epoch"], y=base["activity"], name="actual", line=dict(color="black")))
    for m in shown:
        g = f[(f["cid"] == cid) & (f["model"] == m) & (f["h"] == h)]
        fig.add_trace(go.Scatter(x=g["target_epoch"], y=g["y_pred"], name=m))
    t = float(D["thr"].loc[D["thr"]["cid"] == cid, "threshold"].iloc[0])
    fig.add_hline(y=t, line_dash="dash", line_color="red", annotation_text="critical mass")
    for e in D["events"].loc[D["events"]["cid"] == cid, "epoch"]:
        fig.add_vline(x=int(e), line_color="orange", opacity=0.5)
    fig.update_layout(xaxis_title="epoch", yaxis_title="posts per epoch", height=480)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Orange lines = observed critical-mass events. Forecasts shown at their target epoch.")

with tab4:
    al = D["alerts"]
    st.subheader(f"Early-warning alerts ({len(al)})")
    if al.empty:
        st.info("No alerts were raised in the test period.")
    else:
        fig = px.scatter(al, x="origin", y=al["cid"].astype(str), size="severity", color="severity",
                         labels={"y": "community", "origin": "alert epoch"})
        ev = D["events"]
        fig.add_trace(go.Scatter(x=ev["epoch"], y=ev["cid"].astype(str), mode="markers", name="critical-mass event",
                                 marker=dict(symbol="x", size=12, color="red")))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(al.sort_values("origin"), use_container_width=True, hide_index=True)

with tab5:
    a, b = st.columns(2)
    a.subheader("Accuracy (test period)")
    a.dataframe(D["acc"].round(2), use_container_width=True, hide_index=True)
    a.plotly_chart(px.bar(D["acc"], x="model", y="MAE"), use_container_width=True)
    b.subheader("Early warning")
    b.dataframe(D["ew"].round(2), use_container_width=True, hide_index=True)
    b.plotly_chart(px.bar(D["ew"], x="model", y="recall", range_y=[0, 1]), use_container_width=True)
    st.caption("LT90 = lead time (epochs) reached or exceeded by 90 % of detected events.")
