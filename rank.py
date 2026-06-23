#!/usr/bin/env python3
"""
rank.py
=======
Single entry point that reproduces the submission CSV from the raw
candidate pool.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Architecture (full writeup in README.md and the slide deck):

  1. Stream-parse candidates.jsonl once. For each candidate, extract a
     flat feature dict (features.py) -- title, structured skill
     evidence, career-history-derived trap flags, honeypot flags, and
     behavioral signals. Also build a per-candidate text blob.

  2. Fit one TF-IDF vectorizer over the corpus of all candidate text
     blobs (+ the JD's distilled requirement text), then compute cosine
     similarity between the JD vector and every candidate vector. This
     is the "semantic" component of the score -- fully local, no
     network, no GPU, deterministic.

  3. Combine structured features + semantic similarity into a single
     composite score per candidate (scoring.py): title acts as a
     multiplicative gate, skill/semantic/experience/location/education
     are additively blended, JD-explicit anti-fit traps apply a
     multiplicative penalty, and Redrob behavioral signals apply a
     final multiplicative availability/engagement adjustment. Honeypots
     are forced to (near) zero and hard-excluded.

  4. Sort by score (ties broken by candidate_id ascending, per spec),
     take the top 100, write the CSV in the required format.

Compute profile (measured on a 1-vCPU / 4GB dev sandbox -- comfortably
under the 5-minute / 16GB / CPU-only / no-network budget on real
hardware): see PERFORMANCE.md.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
from datetime import date
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from features import extract_features
from jd_profile import JD_QUERY_TEXT
from scoring import build_reasoning, composite_score

TOP_N = 100


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_and_extract(candidates_path: Path, today: date):
    """Single streaming pass: parse JSON, extract features. Returns a
    list of feature dicts (one per candidate). Supports both JSONL
    (one object per line) and JSON array formats."""
    feats_list = []
    with _open_maybe_gzip(candidates_path) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON array format (e.g. sample_candidates.json)
            records = json.load(f)
            for record in records:
                feats_list.append(extract_features(record, today))
        else:
            # JSONL format (one record per line)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                feats_list.append(extract_features(record, today))
    return feats_list


def compute_semantic_scores(feats_list: list[dict]) -> list[float]:
    """Fit TF-IDF over (JD text + every candidate's doc_text) and return
    cosine similarity of each candidate against the JD vector."""
    corpus = [JD_QUERY_TEXT] + [f["doc_text"] for f in feats_list]
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    jd_vec = matrix[0]
    cand_matrix = matrix[1:]
    sims = cosine_similarity(jd_vec, cand_matrix).ravel()
    return sims.tolist()


def rank_candidates(candidates_path: Path, today: date, top_n: int = TOP_N):
    t0 = time.time()
    feats_list = load_and_extract(candidates_path, today)
    t1 = time.time()
    print(f"[rank.py] loaded + extracted features for {len(feats_list)} candidates "
          f"in {t1 - t0:.1f}s", file=sys.stderr)

    sem_scores = compute_semantic_scores(feats_list)
    t2 = time.time()
    print(f"[rank.py] computed TF-IDF semantic similarity in {t2 - t1:.1f}s", file=sys.stderr)

    scored = []
    n_honeypot = 0
    for feats, sem in zip(feats_list, sem_scores):
        info = composite_score(feats, sem)
        if feats["is_honeypot"]:
            n_honeypot += 1
        reasoning = build_reasoning(feats, info)
        scored.append(
            {
                "candidate_id": feats["candidate_id"],
                "score": info["score"],
                "reasoning": reasoning,
                "is_honeypot": feats["is_honeypot"],
            }
        )

    # Hard safety net: exclude honeypots from being selectable at all,
    # in addition to the near-zero score they already received.
    eligible = [r for r in scored if not r["is_honeypot"]]
    # Round scores to 4dp before sorting so the sort order is consistent
    # with the 4dp values written to CSV (avoids tie-break failures in validator).
    for r in eligible:
        r["score"] = round(r["score"], 4)
    eligible.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    top = eligible[:top_n]
    t3 = time.time()
    print(f"[rank.py] scored all candidates and selected top {top_n} in "
          f"{t3 - t2:.1f}s (honeypots detected & excluded: {n_honeypot})", file=sys.stderr)
    print(f"[rank.py] total wall time: {t3 - t0:.1f}s", file=sys.stderr)
    return top


def write_submission_csv(top: list[dict], out_path: Path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, row in enumerate(top, start=1):
            writer.writerow([row["candidate_id"], rank, f"{row['score']:.4f}", row["reasoning"]])


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob hackathon JD.")
    parser.add_argument("--candidates", required=True, type=Path,
                         help="Path to candidates.jsonl (or .jsonl.gz)")
    parser.add_argument("--out", required=True, type=Path, help="Path to write the output CSV")
    parser.add_argument("--today", default="2026-06-17",
                         help="Reference 'today' date (YYYY-MM-DD) for recency calculations; "
                              "defaults to a fixed value for full reproducibility.")
    args = parser.parse_args()

    today = date.fromisoformat(args.today)
    top = rank_candidates(args.candidates, today, TOP_N)
    write_submission_csv(top, args.out)
    print(f"[rank.py] wrote {len(top)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
