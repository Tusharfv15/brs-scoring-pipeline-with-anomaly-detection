"""
Streamlit UI that orchestrates the four BRS pipeline scripts as subprocesses
and streams their logs live. Produces a downloadable HTML report.

    streamlit run app.py
"""

import os
import sys
import subprocess
from pathlib import Path

import streamlit as st

REPO_ROOT   = Path(__file__).parent.resolve()
EXCEL_DIR   = REPO_ROOT / "excel"
OUTPUTS_DIR = REPO_ROOT / "outputs"

st.set_page_config(page_title="BRS Pipeline", layout="wide")
st.title("BRS — Business Reliability Score Pipeline")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

st.session_state.setdefault("stages", [])          # list[dict]: label, cmd, lines, rc
st.session_state.setdefault("report_bytes", None)  # bytes
st.session_state.setdefault("report_name", None)   # str
st.session_state.setdefault("pipeline_msg", None)  # (level, text)


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Input")

    EXCEL_DIR.mkdir(exist_ok=True)
    existing = sorted(p.name for p in EXCEL_DIR.glob("*.xlsx"))

    source = st.radio("Source", ["Pick existing file", "Upload new file"], index=0)

    selected_path: Path | None = None

    if source == "Pick existing file":
        if not existing:
            st.warning("No .xlsx files found in `excel/`.")
        else:
            chosen = st.selectbox("File", existing)
            selected_path = EXCEL_DIR / chosen
    else:
        uploaded = st.file_uploader("Upload .xlsx", type=["xlsx"])
        if uploaded is not None:
            dest = EXCEL_DIR / uploaded.name
            dest.write_bytes(uploaded.getvalue())
            selected_path = dest
            st.caption(f"Saved to `excel/{uploaded.name}`")

    st.header("Options")
    skip_fetch      = st.checkbox("Skip Places API fetch (`--skip-fetch`)", value=False)
    force_sentiment = st.checkbox("Force re-score sentiment (`--force`)",   value=False)
    override_cat    = st.checkbox("Override business category (`--category`)", value=False)
    category_value  = ""
    if override_cat:
        category_value = st.text_input("Category", placeholder="e.g. Restaurant")

    run_clicked = st.button("Run pipeline", type="primary", disabled=selected_path is None)


# ---------------------------------------------------------------------------
# Stage execution + rendering
# ---------------------------------------------------------------------------

def stream_stage(label: str, cmd: list[str]) -> int:
    """Run a stage live, stream its output into an st.status block, and
    persist the full transcript in st.session_state for post-run reruns."""
    stage = {"label": label, "cmd": cmd, "lines": [], "rc": None}
    st.session_state.stages.append(stage)

    with st.status(label, expanded=True) as status:
        st.caption("`" + " ".join(cmd) + "`")
        log_box     = st.container(height=320)
        placeholder = log_box.empty()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"

        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            stage["lines"].append(line.rstrip("\n"))
            placeholder.code("\n".join(stage["lines"]), language="text")

        stage["rc"] = proc.wait()

        if stage["rc"] == 0:
            status.update(label=f"{label} — done", state="complete", expanded=False)
        else:
            status.update(label=f"{label} — failed (exit {stage['rc']})",
                          state="error", expanded=True)

    return stage["rc"]


def render_past_stage(stage: dict) -> None:
    """Re-render a previously completed stage from session_state (no process)."""
    rc = stage["rc"]
    if rc == 0:
        label, state, expanded = f"{stage['label']} — done", "complete", False
    elif rc is None:
        label, state, expanded = f"{stage['label']} — interrupted", "error", True
    else:
        label, state, expanded = f"{stage['label']} — failed (exit {rc})", "error", True

    with st.status(label, state=state, expanded=expanded):
        st.caption("`" + " ".join(stage["cmd"]) + "`")
        if stage["lines"]:
            with st.container(height=320):
                st.code("\n".join(stage["lines"]), language="text")


def python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, "-u", script, *args]


# ---------------------------------------------------------------------------
# Run pipeline (only when the button was just clicked)
# ---------------------------------------------------------------------------

if run_clicked and selected_path is not None:
    # Reset prior run state
    st.session_state.stages       = []
    st.session_state.report_bytes = None
    st.session_state.report_name  = None
    st.session_state.pipeline_msg = None

    stem     = selected_path.stem
    xlsx_arg = f"excel/{selected_path.name}"

    st.info(f"Running pipeline for **{stem}**")

    # Stage 1 — Clean & Fetch
    s1 = [xlsx_arg]
    if skip_fetch:
        s1.append("--skip-fetch")
    if override_cat and category_value.strip():
        s1.extend(["--category", category_value.strip()])

    ok = stream_stage("Stage 1 — Clean & Fetch (sort_reviews_by_date.py)",
                      python_cmd("sort_reviews_by_date.py", *s1)) == 0

    # Stage 2 — Sentiment
    if ok:
        s2 = [xlsx_arg] + (["--force"] if force_sentiment else [])
        ok = stream_stage("Stage 2 — Sentiment Scoring (sentiment_scorer.py)",
                          python_cmd("sentiment_scorer.py", *s2)) == 0

    # Stage 3 — BRS scoring (always --quiet)
    if ok:
        ok = stream_stage("Stage 3 — BRS Scoring (scorer.py --quiet)",
                          python_cmd("scorer.py", xlsx_arg, "--quiet")) == 0

    # Stage 4 — Anomaly agent
    if ok:
        ok = stream_stage("Stage 4 — Anomaly Detection (anomaly_agent.py)",
                          python_cmd("anomaly_agent.py", xlsx_arg)) == 0

    # Final result
    if not ok:
        st.session_state.pipeline_msg = ("error", "Pipeline stopped — a stage failed. See logs above.")
    else:
        report_path = OUTPUTS_DIR / f"{stem}_report.html"
        if report_path.exists():
            st.session_state.report_bytes = report_path.read_bytes()
            st.session_state.report_name  = report_path.name
            st.session_state.pipeline_msg = ("success", "Pipeline complete.")
        else:
            st.session_state.pipeline_msg = (
                "error",
                f"Pipeline finished but `outputs/{report_path.name}` was not produced.",
            )

else:
    # Not a fresh run — re-render whatever was stored last time so the UI
    # survives reruns caused by the download button, widget changes, etc.
    for stage in st.session_state.stages:
        render_past_stage(stage)


# ---------------------------------------------------------------------------
# Final banner + download button (rendered on every run)
# ---------------------------------------------------------------------------

msg = st.session_state.pipeline_msg
if msg:
    level, text = msg
    {"success": st.success, "error": st.error, "info": st.info}[level](text)

if st.session_state.report_bytes:
    st.download_button(
        label="Download HTML report",
        data=st.session_state.report_bytes,
        file_name=st.session_state.report_name,
        mime="text/html",
        type="primary",
    )
