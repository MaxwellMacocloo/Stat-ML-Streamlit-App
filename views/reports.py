import html
import json
import platform
import sys
from datetime import datetime
from importlib import metadata

import pandas as pd
import streamlit as st

from core.utils import data_fingerprint, get_df

PACKAGES = ["streamlit", "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "lifelines",
            "plotly", "xgboost", "matplotlib"]

CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:980px;margin:40px auto;padding:0 24px;
     color:#1C2430;line-height:1.55}
h1{font-size:2rem;margin-bottom:.2rem}h2{font-size:1.3rem;margin-top:2.4rem;border-top:1px solid #D5DBE3;
     padding-top:1rem;color:#2B4C7E}
.meta{color:#5B6675;font-size:.9rem}pre{background:#F3F5F8;padding:10px;font-size:.8rem;overflow-x:auto}
table{border-collapse:collapse;font-family:Arial,sans-serif;font-size:.82rem;margin:.6rem 0 1.2rem}
th,td{border-bottom:1px solid #E1E6ED;padding:4px 10px;text-align:right}th{background:#EEF1F5}
.wrap{overflow-x:auto}
"""


def versions():
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "not installed"
    return out


def session_info():
    df = get_df()
    return dict(generated=datetime.now().isoformat(timespec="seconds"),
                python=sys.version.split()[0], platform=platform.platform(),
                dataset=st.session_state.get("df_name"), shape=None if df is None else list(df.shape),
                data_fingerprint_md5=data_fingerprint(), packages=versions())


def build_html(entries, title, author, notes):
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{CSS}</style></head><body><h1>{html.escape(title)}</h1>"
             f"<p class='meta'>{html.escape(author)} · generated {datetime.now():%Y-%m-%d %H:%M}</p>"]
    if notes:
        parts.append(f"<p>{html.escape(notes)}</p>")
    first_fig = True
    for i, e in enumerate(entries, 1):
        parts.append(f"<h2>{i}. {html.escape(e['title'])}</h2><p class='meta'>{e['time']} · dataset: "
                     f"{html.escape(str(e['dataset']))} {e['shape']} · fingerprint {e['fingerprint']}</p>")
        parts.append("<pre>" + html.escape(json.dumps(e["params"], indent=2)) + "</pre>")
        for name, tab in e["tables"].items():
            parts.append(f"<h3>{html.escape(name)}</h3><div class='wrap'>"
                         + tab.to_html(float_format=lambda x: f"{x:.4g}", na_rep="—", border=0) + "</div>")
        for fig in e["figs"]:
            parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first_fig else False))
            first_fig = False
    parts.append("<h2>Session information</h2><pre>" + html.escape(json.dumps(session_info(), indent=2))
                 + "</pre></body></html>")
    return "\n".join(parts)


def page_reports():
    st.header("Reports & reproducibility")
    log = st.session_state.get("log", [])
    t1, t2 = st.tabs(["Analysis report", "Reproducibility"])

    with t1:
        if not log:
            st.info("Nothing logged yet. Analyses are added here when you run them "
                    "(or press **Add to report** on the GLM and survival pages).")
        else:
            c1, c2 = st.columns(2)
            title = c1.text_input("Report title", "StatML Lab analysis report")
            author = c2.text_input("Author", "")
            notes = st.text_area("Summary notes (optional)")
            keep = []
            for i, e in enumerate(log):
                c1, c2 = st.columns([1, 12])
                inc = c1.checkbox("Include", True, key=f"inc_{i}", label_visibility="collapsed")
                with c2.expander(f"{i + 1}. {e['title']}  ({e['time']})"):
                    st.caption(f"Dataset: {e['dataset']} {e['shape']} · fingerprint {e['fingerprint']}")
                    st.json(e["params"], expanded=False)
                    for name, tab in e["tables"].items():
                        st.markdown(f"**{name}**")
                        st.dataframe(tab, width="stretch")
                if inc:
                    keep.append(e)
            c1, c2, c3 = st.columns(3)
            c1.download_button("Download HTML report", build_html(keep, title, author, notes).encode(),
                               "statml_report.html", "text/html", type="primary", disabled=not keep)
            payload = [{k: v for k, v in e.items() if k not in ("tables", "figs")} | {
                "tables": {n: json.loads(t.reset_index().to_json(orient="records")) for n, t in e["tables"].items()}}
                for e in keep]
            c2.download_button("Download analysis log (JSON)",
                               json.dumps(dict(session=session_info(), analyses=payload), indent=2, default=str),
                               "statml_log.json", "application/json")
            if c3.button("Clear log"):
                st.session_state.log = []
                st.rerun()

    with t2:
        info = session_info()
        c1, c2, c3 = st.columns(3)
        c1.metric("Python", info["python"])
        c2.metric("Dataset fingerprint", info["data_fingerprint_md5"] or "—",
                  help="MD5 of the active data. Identical fingerprints mean identical data.")
        c3.metric("Analyses logged", len(log))
        st.write(f"**Active dataset:** {info['dataset'] or 'none'} {info['shape'] or ''}")
        st.dataframe(pd.Series(info["packages"], name="version").to_frame(), width="stretch")
        pins = "\n".join(f"{k}=={v}" for k, v in info["packages"].items() if v != "not installed")
        c1, c2 = st.columns(2)
        c1.download_button("Download pinned requirements", pins, "requirements-lock.txt")
        df = get_df()
        if df is not None:
            c2.download_button("Download active data (CSV)", df.to_csv(index=False).encode(),
                               "statml_data.csv", "text/csv")
        st.markdown("Every analysis records its settings and random seed. To reproduce a result: install the "
                    "pinned requirements, load the same data (check the fingerprint), and re-run with the "
                    "logged settings.")
