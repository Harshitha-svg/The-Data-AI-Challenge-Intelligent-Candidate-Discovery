# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

**Challenge**: Rank the top 100 candidates from a 100,000-candidate pool for a Senior AI Engineer role at Redrob AI.

---

## Reproduce in One Command

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
```

Expected output (measured on 1-core / 4 GB RAM sandbox):
```
[rank.py] loaded + extracted features for 100000 candidates in ~33s
[rank.py] computed TF-IDF semantic similarity in ~42s
[rank.py] scored all candidates and selected top 100 in ~1s  (honeypots excluded: 62)
[rank.py] total wall time: ~76s
```

---

## Setup

```bash
pip install -r requirements.txt
# Place candidates.jsonl in ./data/ (not committed — too large for git)
python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv
python validate_submission.py ./outputs/submission.csv   # → "Submission is valid."
```

**Requirements**: Python 3.10+, scikit-learn ≥ 1.4, numpy ≥ 1.26.  
**Compute profile**: CPU-only, < 2 GB RAM peak, < 2 minutes on a 4-core machine.  
No network calls are made during ranking.

---

## Architecture

The system is a **hybrid rule-based + TF-IDF ranker** — fully reproducible, CPU-only, with no external models or APIs.

```
final_score = title_gate
            × blend(skill_score, semantic_score, experience_fit, location_fit, education_bonus)
            × trap_penalty
            × behavioral_multiplier
```

### 1. Title Gate (multiplicative)
All 47 distinct current titles in the dataset are mapped to a gate weight (0.03–1.0).  
Tier-5 elite titles (Staff ML Engineer, Senior AI Engineer, Lead AI Engineer, Senior Applied Scientist, Senior NLP Engineer, Senior ML Engineer, Senior Data Scientist) receive gates 0.90–1.00.  
Non-technical titles (HR Manager, Accountant, Marketing Manager, etc.) receive gates 0.03–0.10.  
This operationalises the JD's explicit note that a "Marketing Manager with AI keywords in skills" is **not** a fit regardless of skill list.

### 2. Skill Score (weight 34%)
Each skill is classified into one of three tiers based on JD alignment:
- **CORE_AI**: vector DBs (Pinecone, Qdrant, Weaviate, Milvus, FAISS, pgvector), retrieval (BM25, Elasticsearch, OpenSearch, Haystack, LlamaIndex), embedding/fine-tuning (Sentence Transformers, LoRA, QLoRA, PEFT), ranking (Learning to Rank), and core ML (Python, PyTorch, scikit-learn, etc.)
- **BROAD_AI**: LLMs, RAG, LangChain, MLOps, Hugging Face, etc.
- **CV_SPEECH**: Computer Vision, CNNs, YOLO, ASR, etc. (lower weight; per JD's explicit CV-without-NLP disqualifier)

A proficiency multiplier (beginner 0.35 → expert 1.0) and a "trust" multiplier (×0.25 if proficiency is advanced/expert but endorsements=0 and duration_months=0, signalling keyword stuffing) adjust the raw sum before a saturating transform bounds it to [0, 1).

### 3. Semantic Score (weight 34%)
A single TF-IDF vectorizer (20 K features, 1–2 grams, sublinear TF) is fit over the JD text plus every candidate's concatenated document text (headline, summary, title, career history descriptions, skill names weighted by proficiency). Cosine similarity between the JD vector and each candidate vector produces a semantic relevance signal that captures phrasing alignment beyond discrete skill matching.

### 4. Experience Fit (weight 15%)
Soft plateau: full credit for 5–9 years (the JD's stated preference), gentle linear decay outside, floor at 0.35. Explicitly not a hard cutoff, per the JD.

### 5. Location Fit (weight 12%)
1.0 for India + Pune/Noida (JD HQ cities); 0.88 for India + other major tech hubs; 0.75/0.55 elsewhere in India depending on `willing_to_relocate`; 0.55/0.40 outside India.

### 6. Education Bonus (weight 5%)
Tier-1 institution → 1.0; otherwise 0.85. Minor signal — the JD does not emphasise pedigree.

### 7. Trap Penalties (multiplicative)
| Trap | Multiplier | JD Source |
|---|---|---|
| Entire career at IT-services firms only | ×0.45 | Explicit JD disqualifier |
| CV/speech skills with zero NLP/IR skills | ×0.55 | Explicit JD disqualifier |
| Average tenure < 18 months (≥3 roles) | ×0.82 | "Title chaser" signal |
| <12 mo LangChain/OpenAI current role, no prior ML background | ×0.55 | Explicit JD disqualifier |
| ≥3 advanced/expert skills with 0 endorsements + 0 duration | ×0.70 | Keyword stuffer |

### 8. Behavioral Multiplier (Redrob signals)
Derived from the 23 `redrob_signals` fields:
- **Availability**: `open_to_work=false` → ×0.55; notice_period ≤30 d → ×1.05; >90 d → ×0.82
- **Recency**: active within 14 days → ×1.12; inactive >120 days → ×0.55
- **Engagement quality**: recruiter_response_rate, interview_completion_rate, offer_acceptance_rate
- **Trust signals**: verified_email / verified_phone / linkedin_connected

Combined multiplier clamped to [0.30, 1.25].

### 9. Honeypot Exclusion
Three disjoint patterns totalling 62 candidates were identified as honeypots (profiles with internally impossible data) and are hard-excluded from selection entirely, with scores set to 1e-6:
1. `is_current` role duration_months grossly exceeds actual elapsed calendar time
2. ≥3 skills with `proficiency: expert` and `duration_months: 0`
3. Certification year > 2026 (e.g. "AWS ML Specialty 2030")

---

## File Layout

```
repo/
├── rank.py                  # CLI entry point — the reproduce command runs this
├── jd_profile.py            # JD reference data: skill sets, title gates, city lists
├── features.py              # Feature extraction from raw candidate JSON
├── scoring.py               # Score computation, penalties, reasoning generation
├── validate_submission.py   # Official organiser validator (unmodified)
├── requirements.txt
├── README.md
├── submission_metadata.yaml
├── data/
│   ├── candidate_schema.json
│   ├── sample_candidates.json
│   └── candidates.jsonl     # ← place here; not committed (487 MB)
├── outputs/
│   └── submission.csv       # Generated output
├── sandbox/
│   └── app.py               # Streamlit demo app
└── tests/
    └── test_scoring.py
```

---

## Sandbox / Demo

A minimal Streamlit app is in `sandbox/app.py`. Upload a JSON or JSONL file with ≤ 100 candidates and it produces a ranked CSV on the fly.

```bash
pip install streamlit
streamlit run sandbox/app.py
```

Or deploy to Streamlit Community Cloud / Hugging Face Spaces in one click.

---

## AI Tool Disclosure

Claude (Anthropic) was used to assist with code architecture, module writing, and dataset analysis during development. All logic, design decisions, and parameter choices were made by the team; the code is fully understood and defensible. No hosted LLM is called at ranking time — the ranker is entirely local and reproducible offline.

---

## Validation

```bash
python validate_submission.py outputs/submission.csv
# Submission is valid.
```
