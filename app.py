"""
Redrob Hackathon — Candidate Ranking Demo
==========================================
Upload a JSON array or JSONL file with ≤ 100 candidates; the app ranks them
and lets you download the resulting CSV.

Run locally:
    pip install streamlit scikit-learn numpy
    streamlit run sandbox/app.py

Deploy to HF Spaces or Streamlit Cloud by pushing this file (and the
repo root modules) to your repo and pointing the platform to this script.
"""

import sys
import json
import io
import csv
from datetime import date
from pathlib import Path

import streamlit as st

# Allow imports from parent directory when running from sandbox/
sys.path.insert(0, str(Path(__file__).parent.parent))

from features import extract_features
from scoring import composite_score, build_reasoning
from rank import compute_semantic_scores

TODAY = date(2026, 6, 17)  # Fixed for reproducibility; change if needed
MAX_CANDIDATES = 100

st.set_page_config(
    page_title="Redrob Candidate Ranker — Demo",
    page_icon="🔎",
    layout="centered",
)

st.title("🔎 Redrob Candidate Ranker — Demo")
st.caption(
    "Upload a JSON array or JSONL file with up to 100 candidates. "
    "The ranker scores and orders them using the same logic as the full submission."
)

uploaded = st.file_uploader(
    "Upload candidates file (.json or .jsonl)",
    type=["json", "jsonl"],
    help="Must be a JSON array [ {...}, {...} ] or one JSON object per line.",
)

if uploaded is not None:
    raw = uploaded.read().decode("utf-8").strip()
    records = []
    try:
        if raw.startswith("["):
            records = json.loads(raw)
        else:
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except json.JSONDecodeError as e:
        st.error(f"Could not parse file: {e}")
        st.stop()

    if len(records) == 0:
        st.warning("No candidate records found in the file.")
        st.stop()

    if len(records) > MAX_CANDIDATES:
        st.warning(f"File contains {len(records)} candidates — only the first {MAX_CANDIDATES} will be ranked.")
        records = records[:MAX_CANDIDATES]

    with st.spinner(f"Ranking {len(records)} candidates…"):
        feats_list = [extract_features(r, TODAY) for r in records]
        sem_scores = compute_semantic_scores(feats_list)

        scored = []
        for feats, sem in zip(feats_list, sem_scores):
            info = composite_score(feats, sem)
            info["score"] = round(info["score"], 4)
            reasoning = build_reasoning(feats, info)
            scored.append({
                "candidate_id": feats["candidate_id"],
                "score": info["score"],
                "reasoning": reasoning,
                "is_honeypot": feats["is_honeypot"],
            })

        eligible = [r for r in scored if not r["is_honeypot"]]
        eligible.sort(key=lambda r: (-r["score"], r["candidate_id"]))
        honeypots = [r for r in scored if r["is_honeypot"]]

    st.success(f"Done! {len(eligible)} eligible candidates ranked; {len(honeypots)} honeypot(s) excluded.")

    # Build CSV in memory
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for i, row in enumerate(eligible, start=1):
        writer.writerow([row["candidate_id"], i, f"{row['score']:.4f}", row["reasoning"]])
    csv_bytes = buf.getvalue().encode("utf-8")

    st.download_button(
        "⬇️  Download ranked CSV",
        data=csv_bytes,
        file_name="ranked_output.csv",
        mime="text/csv",
    )

    # Display table
    st.subheader("Top results")
    display_rows = [
        {
            "Rank": i + 1,
            "Candidate ID": eligible[i]["candidate_id"],
            "Score": eligible[i]["score"],
            "Reasoning": eligible[i]["reasoning"][:120] + "…" if len(eligible[i]["reasoning"]) > 120 else eligible[i]["reasoning"],
        }
        for i in range(min(20, len(eligible)))
    ]
    st.table(display_rows)

    if honeypots:
        with st.expander(f"Excluded honeypots ({len(honeypots)})"):
            for h in honeypots:
                st.markdown(f"- `{h['candidate_id']}`: {h['reasoning'][:150]}")

else:
    st.info("No file uploaded yet. Use the sample_candidates.json from the challenge bundle to try it out.")
    with st.expander("Expected file format"):
        st.code(
            """
// JSON array format:
[
  { "candidate_id": "CAND_0000001", "profile": { ... }, "skills": [...], ... },
  { "candidate_id": "CAND_0000002", ... }
]

// or JSONL (one object per line):
{ "candidate_id": "CAND_0000001", ... }
{ "candidate_id": "CAND_0000002", ... }
""",
            language="json",
        )
