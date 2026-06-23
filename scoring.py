"""
scoring.py
==========
Turns the per-candidate feature dict (features.py) plus a pre-computed
TF-IDF semantic-similarity score into:

  1. A final composite fit score in roughly [0, 1].
  2. A short, specific, human-readable reasoning string built only from
     facts that are actually present in that candidate's record (so it
     cannot hallucinate skills the candidate doesn't have, and it cannot
     degrade into an identical template across candidates).

Design philosophy (see slides / README for the full writeup):

  final_score = title_gate                       # multiplicative gate
              * weighted_blend(skill, semantic,   # additive "how good
                                experience, loc,   # is the match"
                                education)
              * trap_penalty                      # multiplicative,
                                                    # JD-explicit anti-fit
              * behavioral_multiplier              # "are they actually
                                                    # available/engageable"

  Honeypots are forced to (near) zero regardless of the above, and are
  also hard-excluded from the final top-100 as a second safety net.

Title is a *gate*, not just one more additive feature, because the JD
says explicitly: "A candidate who has all the AI keywords listed as
skills but whose title is 'Marketing Manager' is not a fit, no matter
how perfect their skill list looks." A purely additive score lets a
big skill number compensate for a wrong functional role; a
multiplicative gate does not.
"""

from __future__ import annotations

import math
from typing import Any

from jd_profile import DEFAULT_TITLE_GATE, PREFERRED_CITIES, TITLE_GATE, WELCOME_CITIES

HONEYPOT_SCORE = 1e-6


# ---------------------------------------------------------------------------
# Sub-scores
# ---------------------------------------------------------------------------

def experience_fit(years: float) -> float:
    """JD wants 5-9 years, explicitly says it will consider candidates
    outside the band if other signals are strong -- so this is a soft
    plateau, not a hard cutoff."""
    if 5.0 <= years <= 9.0:
        return 1.0
    if years < 5.0:
        return max(0.35, 1.0 - 0.10 * (5.0 - years))
    return max(0.35, 1.0 - 0.06 * (years - 9.0))


def location_fit(city_key: str, country: str, willing_to_relocate: bool) -> float:
    if country != "India":
        return 0.55 if willing_to_relocate else 0.40
    if city_key in PREFERRED_CITIES:
        return 1.0
    if city_key in WELCOME_CITIES:
        return 0.88
    return 0.75 if willing_to_relocate else 0.55


def skill_score(raw: float) -> float:
    """Saturating transform so a handful of strong, trusted core-skill
    hits gets most of the achievable credit, while avoiding an unbounded
    score for candidates who simply list many skills."""
    return 1.0 - math.exp(-raw / 7.0)


def trap_penalty(feats: dict[str, Any]) -> tuple[float, list[str]]:
    penalty = 1.0
    reasons = []
    if feats["consulting_only_trap"]:
        penalty *= 0.45
        reasons.append("entire career history is IT-services/consulting firms only")
    if feats["cv_without_nlp_trap"]:
        penalty *= 0.55
        reasons.append("skill profile is computer-vision/speech only, no NLP/IR depth")
    if feats["title_chaser_trap"]:
        penalty *= 0.82
        reasons.append("short average tenure across roles (job-hopping signal)")
    if feats["recent_ai_only_trap"]:
        penalty *= 0.55
        reasons.append("AI exposure looks confined to a short, recent role with no earlier ML evidence")
    if feats["untrusted_keyword_count"] >= 3:
        penalty *= 0.7
        reasons.append("several high-proficiency skills have zero endorsements/duration backing them")
    return penalty, reasons


def behavioral_multiplier(feats: dict[str, Any]) -> tuple[float, list[str]]:
    """Combine Redrob platform signals into an availability/engagement
    multiplier. Centered at 1.0; can range roughly 0.45-1.18."""
    notes = []
    m = 1.0

    if not feats["open_to_work"]:
        m *= 0.55
        notes.append("not currently flagged open-to-work")

    d = feats["days_since_active"]
    if d <= 14:
        m *= 1.12
    elif d <= 30:
        m *= 1.04
    elif d <= 60:
        m *= 1.0
    elif d <= 120:
        m *= 0.85
        notes.append(f"hasn't been active on the platform in {d} days")
    else:
        m *= 0.55
        notes.append(f"inactive for {d}+ days")

    rr = feats["recruiter_response_rate"]
    m *= (0.75 + 0.45 * rr)
    if rr < 0.2:
        notes.append(f"low recruiter response rate ({rr:.0%})")

    ic = feats["interview_completion_rate"]
    m *= (0.9 + 0.15 * ic)

    npd = feats["notice_period_days"]
    if npd <= 30:
        m *= 1.05
    elif npd <= 60:
        m *= 1.0
    elif npd <= 90:
        m *= 0.92
        notes.append(f"{npd}-day notice period")
    else:
        m *= 0.82
        notes.append(f"long {npd}-day notice period")

    oar = feats["offer_acceptance_rate"]
    if oar != -1 and oar < 0.15:
        m *= 0.93

    verified = sum([feats["verified_email"], feats["verified_phone"], feats["linkedin_connected"]])
    m *= (0.97 + 0.01 * verified)

    return max(0.30, min(1.25, m)), notes


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

W_SKILL = 0.34
W_SEMANTIC = 0.34
W_EXPERIENCE = 0.15
W_LOCATION = 0.12
W_EDUCATION = 0.05


def composite_score(feats: dict[str, Any], semantic_sim: float) -> dict[str, Any]:
    if feats["is_honeypot"]:
        return {
            "score": HONEYPOT_SCORE,
            "title_gate": 0.0,
            "trap_penalty": 0.0,
            "behavioral_multiplier": 0.0,
            "trap_reasons": [],
            "behavioral_notes": [],
        }

    title_gate = TITLE_GATE.get(feats["title"], DEFAULT_TITLE_GATE)

    sk = skill_score(feats["skill_evidence_raw"])
    sem = max(0.0, min(1.0, semantic_sim))
    exp = experience_fit(feats["years_experience"])
    loc = location_fit(feats["city_key"], feats["country"], feats["willing_to_relocate"])
    edu = 1.0 if feats["has_tier1_education"] else 0.85

    base = (
        W_SKILL * sk
        + W_SEMANTIC * sem
        + W_EXPERIENCE * exp
        + W_LOCATION * loc
        + W_EDUCATION * edu
    )

    gated = title_gate * base
    pen, trap_reasons = trap_penalty(feats)
    trapped = gated * pen
    beh_mult, beh_notes = behavioral_multiplier(feats)
    final = trapped * beh_mult

    return {
        "score": max(0.0, min(1.0, final)),
        "title_gate": title_gate,
        "skill_score": sk,
        "semantic_score": sem,
        "experience_fit": exp,
        "location_fit": loc,
        "trap_penalty": pen,
        "trap_reasons": trap_reasons,
        "behavioral_multiplier": beh_mult,
        "behavioral_notes": beh_notes,
    }


# ---------------------------------------------------------------------------
# Reasoning text generation
# ---------------------------------------------------------------------------

def build_reasoning(feats: dict[str, Any], score_info: dict[str, Any]) -> str:
    if feats["is_honeypot"]:
        return (
            "Excluded: profile contains an internally inconsistent / "
            f"impossible detail ({', '.join(feats['honeypot_reasons'])})."
        )

    title = feats["title"]
    company = feats["company"]
    yrs = feats["years_experience"]

    skill_bits = []
    if feats["core_ai_skill_hits"]:
        skill_bits.append(", ".join(feats["core_ai_skill_hits"][:4]))
    elif feats["broad_ai_skill_hits"]:
        skill_bits.append(", ".join(feats["broad_ai_skill_hits"][:4]))

    lead = f"{title} ({yrs:.1f} yrs) at {company}"
    if skill_bits:
        lead += f"; shows {skill_bits[0]} in their skill/experience record"
    else:
        lead += "; little direct embeddings/retrieval/ranking evidence"

    caveats = score_info.get("trap_reasons", []) + score_info.get("behavioral_notes", [])
    if caveats:
        tail = "Caveat: " + "; ".join(caveats[:2]) + "."
    else:
        rr = feats["recruiter_response_rate"]
        tail = f"Active on platform, recruiter response rate {rr:.0%}, open to work."

    sentence = f"{lead}. {tail}"
    # Keep it tight; CSV reasoning column is meant to be a short justification.
    return sentence[:300]
