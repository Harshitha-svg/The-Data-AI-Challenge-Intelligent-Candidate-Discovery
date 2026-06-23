"""
jd_profile.py
=============
Static, hand-distilled representation of the released Job Description
(Senior AI Engineer - Founding Team, Redrob AI) plus the closed vocabularies
observed in the candidate pool.

Why this file exists
---------------------
The JD is long, conversational, and deliberately includes "trap" language
("we don't care which keywords you have"). Rather than re-parsing the JD
text at runtime, we distill it once into:

  1. A dense "ideal profile" paragraph used as the query side of the
     TF-IDF semantic match (jd_query_text).
  2. Explicit title/skill tiers, derived empirically from the candidate
     pool (see notebooks/eda.md for the exploration that produced these
     tiers) rather than guessed from general knowledge -- the dataset
     uses a *closed* vocabulary of 47 titles and 133 skills, so tiering
     them once, by hand, after inspection is more reliable than fuzzy
     text matching alone.

This file contains NO scoring logic -- only reference data -- so it is
easy for a reviewer (or interviewer) to audit "what does this system
believe about the JD" independent of "how does it compute scores."
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Dense JD text for semantic (TF-IDF) matching.
#    Written to mirror the JD's own emphasis: embeddings/retrieval/ranking,
#    vector databases, evaluation rigor, production shipping, NLP/IR depth,
#    and the explicit "ideal candidate" paragraph from the JD itself.
# ---------------------------------------------------------------------------
JD_QUERY_TEXT = """
Senior AI Engineer founding team role owning the intelligence layer of a
recruiting platform: ranking, retrieval, and matching systems that decide
what recruiters see when they search for candidates. Production experience
with embeddings based retrieval systems, sentence transformers, OpenAI
embeddings, BGE, E5 embedding models deployed to real users, handling
embedding drift, index refresh, retrieval quality regression in production.
Production experience with vector databases and hybrid search infrastructure:
Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS,
pgvector. Strong Python and code quality. Hands on experience designing
evaluation frameworks for ranking systems: NDCG, MRR, MAP, offline to online
correlation, A/B test interpretation. Hybrid retrieval combining dense and
sparse signals such as BM25. Learning to rank models, XGBoost based or
neural ranking, recommendation systems, semantic search, information
retrieval, search and discovery, search infrastructure, indexing algorithms,
ranking systems shipped end to end at a real product company at meaningful
scale, not only in research. LLM fine tuning experience LoRA QLoRA PEFT
prompt engineering retrieval augmented generation RAG LangChain Hugging Face
Transformers LlamaIndex Haystack. Distributed systems and large scale
inference optimization, MLOps, Kubernetes, Docker, model serving,
monitoring, drift detection, feature engineering, feature pipelines.
Hands on engineer who ships working systems quickly, comfortable owning
search ranking or recommendation systems end to end at a product company,
not a pure researcher, not a framework tutorial writer, not someone who has
only called a hosted large language model API for under a year, not someone
who has moved away from writing code into pure architecture or tech lead
work, not someone whose whole career has been at IT services or consulting
companies, not someone whose primary expertise is computer vision speech or
robotics without natural language processing or information retrieval
exposure, not a title chaser switching companies every year and a half.
Six to eight years total experience, four to five years in applied machine
learning or AI roles at product companies, has shipped at least one end to
end ranking search or recommendation system to real users at meaningful
scale, has strong opinions about retrieval hybrid versus dense, evaluation
offline versus online, and large language model integration fine tune versus
prompt, can defend these choices with reference to systems actually built.
Located in or willing to relocate to Noida or Pune, open to Hyderabad Mumbai
Bangalore Delhi NCR Chennai. Short notice period under thirty days preferred.
Mentors junior engineers, works async first, writes clearly, ships in weeks
not quarters.
""".strip()

# ---------------------------------------------------------------------------
# 2. Title tiers (closed vocabulary of 47 distinct `current_title` values
#    observed across the full 100,000-candidate pool). Multiplicative gate:
#    a perfect skill list cannot overcome a structurally wrong functional
#    title, per the JD's explicit instruction.
# ---------------------------------------------------------------------------
TITLE_GATE = {
    # Tier 5 -- exact / near-exact role match. Tiny pool by design (the JD
    # says explicitly it expects very few true matches in 100K candidates).
    "Senior AI Engineer": 1.00,
    "Lead AI Engineer": 1.00,
    "Senior Applied Scientist": 0.97,
    "Staff Machine Learning Engineer": 0.97,
    "Senior Machine Learning Engineer": 0.95,
    "Senior NLP Engineer": 0.95,
    # Tier 4 -- strong applied AI/ML/search roles, one step below "senior".
    "Senior Data Scientist": 0.85,
    "NLP Engineer": 0.85,
    "AI Engineer": 0.83,
    "Search Engineer": 0.85,
    "Applied ML Engineer": 0.83,
    "Machine Learning Engineer": 0.82,
    "Recommendation Systems Engineer": 0.85,
    # Tier 3.5 -- applied AI roles, but junior / narrower / needs evidence check.
    "AI Specialist": 0.65,
    "Senior Software Engineer (ML)": 0.78,
    "Data Scientist": 0.62,
    "ML Engineer": 0.68,
    "Computer Vision Engineer": 0.45,   # JD explicit concern: CV w/o NLP/IR
    "Junior ML Engineer": 0.45,         # explicitly junior
    # Tier 3 -- data/software engineering, plausible adjacent background.
    "Senior Data Engineer": 0.42,
    "Senior Software Engineer": 0.40,
    "Analytics Engineer": 0.35,
    "Data Engineer": 0.35,
    "Data Analyst": 0.30,
    "Backend Engineer": 0.32,
    # Tier 2 -- generic software/infra roles, weak fit unless strong evidence.
    "Software Engineer": 0.22,
    "Full Stack Developer": 0.18,
    "Cloud Engineer": 0.18,
    "DevOps Engineer": 0.16,
    "Java Developer": 0.15,
    ".NET Developer": 0.12,
    "Mobile Developer": 0.12,
    "Frontend Engineer": 0.14,
    "QA Engineer": 0.12,
    # Tier 1 -- non-technical roles. The JD is explicit that a perfect
    # keyword list on a "Marketing Manager" profile is not a fit -- gate
    # hard regardless of skills text.
    "Business Analyst": 0.06,
    "HR Manager": 0.04,
    "Mechanical Engineer": 0.04,
    "Accountant": 0.03,
    "Project Manager": 0.10,
    "Customer Support": 0.03,
    "Operations Manager": 0.06,
    "Content Writer": 0.04,
    "Sales Executive": 0.03,
    "Civil Engineer": 0.03,
    "Graphic Designer": 0.04,
    "Marketing Manager": 0.05,
}
DEFAULT_TITLE_GATE = 0.20  # fallback for any title not seen above

# ---------------------------------------------------------------------------
# 3. Skill vocabulary tiers (133 distinct skill names observed in the pool).
#    Used as part of evidence-weighting (see scoring.py) -- rarity in the
#    observed corpus is also captured automatically via TF-IDF/IDF weighting
#    of free text, but skills are a *structured* field, so we tier them
#    explicitly for the interpretable, auditable part of the score.
# ---------------------------------------------------------------------------
CORE_AI_SKILLS = {
    # Rare (~1.3% prevalence), high-signal: vector DBs, fine-tuning, core
    # ML frameworks/concepts, retrieval-specific terms.
    "QLoRA", "pgvector", "Weaviate", "Milvus", "Learning to Rank", "BM25",
    "TensorFlow", "Qdrant", "Python", "PyTorch", "PEFT", "LoRA", "NLP",
    "Machine Learning", "Deep Learning", "Haystack", "Elasticsearch",
    "LlamaIndex", "scikit-learn", "OpenSearch",
    # Ultra-rare "elite" vocabulary (<10 occurrences, seeded almost
    # exclusively into Tier 5 titles) -- mirrors the JD's own phrasing.
    "Information Retrieval Systems", "Search Backend", "Text Encoders",
    "Vector Representations", "Content Matching", "Model Adaptation",
    "Ranking Systems", "Search & Discovery", "Workflow Orchestration",
    "Search Infrastructure", "Indexing Algorithms", "Open-source ML libraries",
    "Natural Language Processing", "Document Processing",
}

BROAD_AI_SKILLS = {
    # ~5% prevalence -- broader AI/ML vocabulary, mixes genuinely relevant
    # (RAG, Embeddings, Semantic Search) with CV/Speech-only skills.
    "Hugging Face Transformers", "LangChain", "Information Retrieval",
    "LLMs", "Recommendation Systems", "Semantic Search",
    "Sentence Transformers", "Embeddings", "Vector Search",
    "Prompt Engineering", "Pinecone", "FAISS", "RAG", "Fine-tuning LLMs",
    "Feature Engineering", "Data Science", "Reinforcement Learning",
    "Kubeflow", "MLOps", "BentoML", "MLflow", "Weights & Biases",
    "Statistical Modeling", "Time Series", "Forecasting",
}

CV_SPEECH_SKILLS = {
    # JD explicit concern: CV/Speech/robotics expertise without NLP/IR.
    "YOLO", "GANs", "OpenCV", "ASR", "Image Classification",
    "Computer Vision", "Speech Recognition", "CNN", "Object Detection",
    "Diffusion Models", "TTS",
}

GENERIC_TECH_SKILLS_NOTE = (
    "The remaining ~85 skill names (HTML, Excel, Salesforce CRM, Six Sigma, "
    "Tailwind, Java, Kafka, etc., ~12% prevalence each) are generic "
    "software/business skills, treated as low-weight background signal "
    "only -- they neither help nor hurt the AI-fit score directly, but "
    "contribute a small 'is this a working engineer at all' baseline via "
    "the TF-IDF text similarity."
)

# Companies that are 100%-consulting-only-disqualifying when they make up
# the candidate's *entire* career history (per the JD's explicit list).
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "mphasis", "tech mahindra",
}

# Tier-1 / explicitly-welcomed Indian cities (JD section "On location").
PREFERRED_CITIES = {"pune", "noida"}
WELCOME_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon",
    "bangalore", "chennai",
}
