"""
Unit tests for features.py and scoring.py.
Run: python -m pytest tests/
"""
import sys
import json
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from features import extract_features
from scoring import (
    experience_fit, location_fit, skill_score,
    trap_penalty, behavioral_multiplier, composite_score,
    HONEYPOT_SCORE,
)
from jd_profile import CORE_AI_SKILLS

TODAY = date(2026, 6, 17)

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_candidate(**overrides):
    """Minimal valid candidate record."""
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "headline": "Test candidate",
            "summary": "Test summary",
            "location": "Pune, India",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior AI Engineer",
            "current_company": "Test Corp",
            "current_company_size": "11-50",
            "current_industry": "Technology",
        },
        "career_history": [
            {
                "company": "Test Corp",
                "title": "Senior AI Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 36,
                "is_current": True,
                "industry": "Technology",
                "company_size": "11-50",
                "description": "Built vector search and retrieval systems.",
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2012,
                "end_year": 2016,
                "grade": "9.0",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Elasticsearch", "proficiency": "expert", "endorsements": 5, "duration_months": 30},
            {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
            {"name": "Qdrant", "proficiency": "advanced", "endorsements": 2, "duration_months": 18},
        ],
        "certifications": [],
        "languages": [{"language": "English", "proficiency": "fluent"}],
        "redrob_signals": {
            "profile_completeness_score": 0.9,
            "signup_date": "2024-01-01",
            "last_active_date": "2026-06-15",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 4.0,
            "skill_assessment_scores": {"Elasticsearch": 85},
            "connection_count": 300,
            "endorsements_received": 20,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 30, "max": 50},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 75,
            "search_appearance_30d": 15,
            "saved_by_recruiters_30d": 3,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.8,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    base.update(overrides)
    return base


# ── experience_fit ────────────────────────────────────────────────────────────

def test_experience_fit_in_range():
    assert experience_fit(7.0) == 1.0

def test_experience_fit_below():
    assert experience_fit(2.0) < 1.0
    assert experience_fit(2.0) > 0.3

def test_experience_fit_above():
    # Slightly above is still high
    assert experience_fit(10.0) >= 0.85


# ── location_fit ──────────────────────────────────────────────────────────────

def test_location_fit_pune():
    assert location_fit("pune", "India", True) == 1.0

def test_location_fit_noida():
    assert location_fit("noida", "India", True) == 1.0

def test_location_fit_mumbai():
    score = location_fit("mumbai", "India", True)
    assert 0.8 < score < 1.0

def test_location_fit_outside_india():
    score = location_fit("san francisco", "USA", False)
    assert score < 0.6


# ── skill_score ───────────────────────────────────────────────────────────────

def test_skill_score_core_skills():
    # Strong core AI skills should produce a high score
    core_skills = [
        {"name": "Elasticsearch", "proficiency": "expert", "endorsements": 5, "duration_months": 24},
        {"name": "Qdrant", "proficiency": "expert", "endorsements": 3, "duration_months": 18},
        {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 48},
        {"name": "BM25", "proficiency": "advanced", "endorsements": 2, "duration_months": 12},
    ]
    feats = extract_features(_make_candidate(**{"skills": core_skills}), TODAY)
    score = skill_score(feats["skill_evidence_raw"])
    assert score > 0.7

def test_skill_score_untrusted():
    # Expert skills with 0 endorsements and 0 duration should be penalised
    untrusted = [
        {"name": "Elasticsearch", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Qdrant", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
        {"name": "Python", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
    ]
    trusted = [
        {"name": "Elasticsearch", "proficiency": "expert", "endorsements": 5, "duration_months": 24},
        {"name": "Qdrant", "proficiency": "expert", "endorsements": 3, "duration_months": 18},
        {"name": "Python", "proficiency": "expert", "endorsements": 10, "duration_months": 36},
    ]
    feats_untrusted = extract_features(_make_candidate(**{"skills": untrusted}), TODAY)
    feats_trusted = extract_features(_make_candidate(**{"skills": trusted}), TODAY)
    assert skill_score(feats_trusted["skill_evidence_raw"]) > skill_score(feats_untrusted["skill_evidence_raw"])


# ── trap_penalty ──────────────────────────────────────────────────────────────

def test_trap_penalty_consulting_only():
    consulting_history = [
        {"company": "TCS", "title": "Software Engineer", "start_date": "2018-01-01",
         "end_date": "2021-01-01", "duration_months": 36, "is_current": False,
         "industry": "IT Services", "company_size": "10000+", "description": "Worked on projects."},
        {"company": "Infosys", "title": "Senior Engineer", "start_date": "2021-01-01",
         "end_date": None, "duration_months": 60, "is_current": True,
         "industry": "IT Services", "company_size": "10000+", "description": ""},
    ]
    record = _make_candidate(career_history=consulting_history)
    record["profile"]["current_company"] = "Infosys"
    feats = extract_features(record, TODAY)
    penalty, _ = trap_penalty(feats)
    assert penalty < 0.5  # consulting-only trap should heavily penalise

def test_trap_penalty_no_traps():
    feats = extract_features(_make_candidate(), TODAY)
    penalty, reasons = trap_penalty(feats)
    assert penalty == 1.0
    assert reasons == []


# ── honeypot exclusion ────────────────────────────────────────────────────────

def test_honeypot_future_cert():
    record = _make_candidate()
    record["certifications"] = [{"name": "AWS ML Specialty", "issuer": "AWS", "year": 2030}]
    feats = extract_features(record, TODAY)
    assert feats["is_honeypot"] is True
    info = composite_score(feats, 0.5)
    assert info["score"] == HONEYPOT_SCORE

def test_honeypot_multiple_expert_zero_duration():
    record = _make_candidate()
    record["skills"] = [
        {"name": s, "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for s in ["Elasticsearch", "Python", "Qdrant", "BM25", "Weaviate"]
    ]
    feats = extract_features(record, TODAY)
    assert feats["is_honeypot"] is True


# ── composite_score sanity ────────────────────────────────────────────────────

def test_composite_score_elite_candidate():
    feats = extract_features(_make_candidate(), TODAY)
    info = composite_score(feats, 0.7)  # strong semantic match
    assert info["score"] > 0.7

def test_composite_score_bad_title():
    record = _make_candidate()
    record["profile"]["current_title"] = "Marketing Manager"
    feats = extract_features(record, TODAY)
    info = composite_score(feats, 0.7)
    assert info["score"] < 0.15  # title gate should crush the score
