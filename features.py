"""
features.py
============
Pure functions that turn one raw candidate JSON record (as loaded from
candidates.jsonl) into a flat dict of interpretable features. No scoring
happens here -- this module only extracts evidence. scoring.py turns
evidence into numbers.

Kept dependency-free (stdlib only) so it is trivial to unit test and to
reproduce in any sandbox.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from jd_profile import (
    CONSULTING_FIRMS,
    CORE_AI_SKILLS,
    BROAD_AI_SKILLS,
    CV_SPEECH_SKILLS,
    PREFERRED_CITIES,
    WELCOME_CITIES,
)

PROFICIENCY_WEIGHT = {
    "beginner": 0.35,
    "intermediate": 0.6,
    "advanced": 0.85,
    "expert": 1.0,
}


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _month_diff(d1: date, d2: date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def _city_key(location: str) -> str:
    """First comma-segment of location, lowercased, for city-set lookups."""
    return location.split(",")[0].strip().lower()


def extract_features(record: dict[str, Any], today: date) -> dict[str, Any]:
    """Extract a flat feature dict from one candidate record.

    `today` is passed in explicitly (rather than using datetime.now())
    so that feature extraction is fully deterministic and reproducible
    regardless of when the ranking script is actually run.
    """
    cid = record["candidate_id"]
    profile = record["profile"]
    career = record.get("career_history", []) or []
    education = record.get("education", []) or []
    skills = record.get("skills", []) or []
    certs = record.get("certifications", []) or []
    signals = record.get("redrob_signals", {}) or {}

    feats: dict[str, Any] = {"candidate_id": cid}

    # ---- profile basics -----------------------------------------------
    feats["title"] = profile.get("current_title", "")
    feats["company"] = profile.get("current_company", "")
    feats["years_experience"] = float(profile.get("years_of_experience", 0.0))
    feats["location"] = profile.get("location", "")
    feats["country"] = profile.get("country", "")
    feats["city_key"] = _city_key(feats["location"])

    # ---- text blob for semantic similarity -----------------------------
    text_parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        feats["title"],
    ]
    for c in career:
        text_parts.append(c.get("title", ""))
        text_parts.append(c.get("description", ""))
    # Weight skills into the text blob proportionally to proficiency, so
    # TF-IDF naturally up-weights candidates with deep (not just present)
    # skill claims, and naturally up-weights rare/specific skill terms via
    # corpus-wide IDF.
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "beginner")
        repeats = {"beginner": 1, "intermediate": 1, "advanced": 2, "expert": 3}.get(prof, 1)
        text_parts.extend([name] * repeats)
    feats["doc_text"] = " ".join(p for p in text_parts if p)

    # ---- skill evidence (structured, auditable) -------------------------
    core_hits, broad_hits, cv_hits = [], [], []
    skill_evidence_score = 0.0
    untrusted_keyword_count = 0  # expert/advanced claim w/ 0 endorsements & 0 duration
    for s in skills:
        name = s.get("name", "")
        prof = s.get("proficiency", "beginner")
        endorsements = s.get("endorsements", 0) or 0
        duration_m = s.get("duration_months", 0) or 0
        prof_w = PROFICIENCY_WEIGHT.get(prof, 0.35)

        is_untrusted = prof in ("advanced", "expert") and endorsements == 0 and duration_m == 0
        trust = 0.25 if is_untrusted else 1.0
        if is_untrusted:
            untrusted_keyword_count += 1

        if name in CORE_AI_SKILLS:
            weight = 3.0
            core_hits.append(name)
        elif name in BROAD_AI_SKILLS:
            weight = 1.5
            broad_hits.append(name)
        elif name in CV_SPEECH_SKILLS:
            weight = 1.0
            cv_hits.append(name)
        else:
            weight = 0.0  # generic skill, no AI-fit weight

        skill_evidence_score += weight * prof_w * trust

    feats["core_ai_skill_hits"] = core_hits
    feats["broad_ai_skill_hits"] = broad_hits
    feats["cv_speech_skill_hits"] = cv_hits
    feats["skill_evidence_raw"] = skill_evidence_score
    feats["untrusted_keyword_count"] = untrusted_keyword_count

    # CV/Speech-without-NLP/IR trap: dominated by CV/speech skills, zero
    # core or broad NLP/IR skill evidence at all.
    feats["cv_without_nlp_trap"] = bool(cv_hits) and not core_hits and not broad_hits

    # ---- career history derived signals ---------------------------------
    n_roles = len(career)
    feats["n_roles"] = n_roles
    durations = [c.get("duration_months", 0) or 0 for c in career]
    feats["avg_tenure_months"] = (sum(durations) / n_roles) if n_roles else 0.0
    feats["title_chaser_trap"] = n_roles >= 3 and feats["avg_tenure_months"] < 18

    companies_lower = [c.get("company", "").lower() for c in career]
    companies_lower.append(feats["company"].lower())

    def _is_consulting(name: str) -> bool:
        return any(k in name for k in CONSULTING_FIRMS)

    feats["consulting_only_trap"] = bool(companies_lower) and all(
        _is_consulting(c) for c in companies_lower
    )

    # Recent-AI-only trap: AI-related skill text appears only in the
    # current role, that role is short (<12 months), and no other role's
    # description shows pre-existing ML/data evidence.
    ai_text_pattern = re.compile(
        r"langchain|openai|llm|rag\b|prompt engineer|gpt|embedding",
        re.IGNORECASE,
    )
    pre_existing_ml_pattern = re.compile(
        r"machine learning|model|pipeline|ranking|recommendation|nlp|"
        r"data scien|forecast|classif|regression|feature engineer|"
        r"vector|search|retrieval|deep learning|neural",
        re.IGNORECASE,
    )
    current_role = next((c for c in career if c.get("is_current")), None)
    other_roles = [c for c in career if c is not current_role]
    recent_ai_only = False
    if current_role is not None:
        cur_text = current_role.get("description", "")
        cur_is_short = (current_role.get("duration_months", 0) or 0) < 12
        cur_has_ai = bool(ai_text_pattern.search(cur_text))
        has_other_ml_evidence = any(
            pre_existing_ml_pattern.search(r.get("description", "")) for r in other_roles
        ) or any(s.get("name") in CORE_AI_SKILLS for s in skills)
        if cur_has_ai and cur_is_short and not has_other_ml_evidence:
            recent_ai_only = True
    feats["recent_ai_only_trap"] = recent_ai_only

    # ---- education ---------------------------------------------------
    tiers = [e.get("tier", "unknown") for e in education]
    feats["has_tier1_education"] = "tier_1" in tiers

    # ---- behavioral signals (redrob_signals) -----------------------
    feats["open_to_work"] = bool(signals.get("open_to_work_flag", False))
    last_active = _parse_date(signals.get("last_active_date"))
    feats["days_since_active"] = (today - last_active).days if last_active else 9999
    feats["recruiter_response_rate"] = float(signals.get("recruiter_response_rate", 0.0) or 0.0)
    feats["interview_completion_rate"] = float(signals.get("interview_completion_rate", 0.0) or 0.0)
    feats["notice_period_days"] = int(signals.get("notice_period_days", 90) or 90)
    feats["profile_completeness"] = float(signals.get("profile_completeness_score", 0.0) or 0.0)
    feats["willing_to_relocate"] = bool(signals.get("willing_to_relocate", False))
    feats["verified_email"] = bool(signals.get("verified_email", False))
    feats["verified_phone"] = bool(signals.get("verified_phone", False))
    feats["linkedin_connected"] = bool(signals.get("linkedin_connected", False))
    feats["saved_by_recruiters_30d"] = int(signals.get("saved_by_recruiters_30d", 0) or 0)
    feats["search_appearance_30d"] = int(signals.get("search_appearance_30d", 0) or 0)
    feats["github_activity_score"] = float(signals.get("github_activity_score", -1) or -1)
    feats["offer_acceptance_rate"] = float(signals.get("offer_acceptance_rate", -1) or -1)
    feats["avg_response_time_hours"] = float(signals.get("avg_response_time_hours", 999) or 999)

    # ---- honeypot detectors (subtly impossible profiles) -----------
    honeypot_reasons = []

    # (a) a current ("is_current": true) role claims a duration_months
    # far longer than the actual elapsed time since its start_date.
    for c in career:
        if not c.get("is_current"):
            continue
        sd = _parse_date(c.get("start_date"))
        if sd is None:
            continue
        elapsed = _month_diff(sd, today)
        claimed = c.get("duration_months", 0) or 0
        if elapsed > 0 and claimed > elapsed * 2 and (claimed - elapsed) > 24:
            honeypot_reasons.append("current_role_duration_exceeds_elapsed_time")
            break

    # (b) 3+ skills claiming "expert" proficiency with literally zero
    # months of use -- "expert with 0 years used".
    n_expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months", 0) or 0) == 0
    )
    if n_expert_zero >= 3:
        honeypot_reasons.append("multiple_expert_skills_zero_duration")

    # (c) a certification dated in the future relative to `today`.
    for c in certs:
        yr = c.get("year")
        if isinstance(yr, int) and yr > today.year:
            honeypot_reasons.append("future_dated_certification")
            break

    feats["honeypot_reasons"] = honeypot_reasons
    feats["is_honeypot"] = len(honeypot_reasons) > 0

    return feats
