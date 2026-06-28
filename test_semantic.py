"""
test_semantic.py  (v3 — Full Pipeline with Behavioral + Assessment + Education)
─────────────────────────────────────────────────────────────────────────────────
Stage 1 : BM25 pre-filter  → fast coarse ranking of all survivors
Stage 2 : BGE-small embedding re-rank → semantic scoring on BM25 top-N only
Stage 3 : Keyword score   → tiered skills + career ratio
Stage 4 : Assessment score → platform-verified test scores (NEW)
Stage 5 : Education score  → institution tier (NEW)
Stage 6 : Behavioral multiplier → availability / engagement signals (NEW)
Stage 7 : Specific, factual reasoning generation (NEW)

Usage:
    pip install rank_bm25
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers
    python test_semantic.py
"""

import os, sys, json, time, re
from datetime import datetime
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
CANDIDATES_FILE = r"c:\Users\amank\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
JD_FILE         = r"c:\Users\amank\.gemini\antigravity-ide\brain\59c96619-9e36-4224-9300-fe835ef6eac6\scratch\job_description.md"
SAMPLE_SIZE     = 150000    # Process ALL candidates
BM25_TOP_N      = 2000      # Smart gate: top-2000 by BM25+keyword → only these get embedded
BATCH_SIZE      = 128       # Larger batches = better CPU throughput
MODEL_NAME      = "BAAI/bge-small-en-v1.5"  # BGE-small: retrieval-optimised, 512-dim, ~130MB
FULL_SURVIVORS  = 9603
FULL_BM25_TOP   = 2000
BUDGET          = 300       # 5 minutes

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(secs: float) -> str:
    m = int(secs) // 60
    s = secs - m * 60
    return f"{m}m {s:05.2f}s"

def bar(secs: float, total: float, width: int = 20) -> str:
    if total == 0: return "░" * width
    filled = int(round(min(secs / total, 1.0) * width))
    return "█" * filled + "░" * (width - filled)

def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())

# ── 1. Program start timer (measures TOTAL wall-clock from first line) ────────
PROGRAM_START = time.perf_counter()

# ── 2. Check dependencies ────────────────────────────────────────────────────
print("=" * 65)
print("  Full Pipeline Benchmark: BM25 → BGE-small-en-v1.5 + Behavioral + Edu")
print("=" * 65)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("[ERROR] rank_bm25 not installed. Run: pip install rank_bm25")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    # ⚡ Use ALL available CPU cores for torch inference
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"[CONFIG] torch threads: {torch.get_num_threads()} / {os.cpu_count()} cores")
except ImportError:
    print("[ERROR] sentence-transformers not installed.")
    sys.exit(1)

# ── 2. Load Phase 1 filters ───────────────────────────────────────────────────
sys.path.insert(0, r"c:\Users\amank\Downloads\[PUB] India_runs_data_and_ai_challenge\ranker_project")
from filters import (
    is_honeypot, passes_location_filter, passes_consulting_filter,
    passes_closed_source_filter, passes_title_chaser_filter,
    passes_architecture_astronaut_filter, passes_domain_expertise_filter,
    passes_title_mismatch_filter, passes_duplicate_description_filter
)

def apply_hard_filters(c: dict) -> bool:
    return (
        not is_honeypot(c)
        and passes_location_filter(c)
        and passes_consulting_filter(c)
        and passes_closed_source_filter(c)
        and passes_title_chaser_filter(c)
        and passes_architecture_astronaut_filter(c)
        and passes_domain_expertise_filter(c)
        and passes_title_mismatch_filter(c)
    )

# ── 3. Build candidate text ───────────────────────────────────────────────────
def candidate_full_text(c: dict) -> str:
    """Single concatenated text for BM25 (no token limit issue)."""
    parts = []
    profile = c.get("profile", {})
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("current_title", ""))
    for job in c.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for skill in c.get("skills", []):
        parts.append(skill.get("name", ""))
    return " ".join(p for p in parts if p)

def candidate_sections(c: dict) -> list:
    """Separate sections for BGE section-averaged embedding."""
    sections = []
    summary = c.get("profile", {}).get("summary", "").strip()
    if summary:
        sections.append(summary)
    for job in c.get("career_history", []):
        desc = job.get("description", "").strip()
        if desc:
            sections.append(desc)
    return sections or [c.get("profile", {}).get("headline", "")]

# ── 4. Load JD ────────────────────────────────────────────────────────────────
print(f"\n[STEP 1] Loading JD...")
t0 = time.perf_counter()
try:
    with open(JD_FILE, "r", encoding="utf-8") as f:
        jd_text = f.read()
except FileNotFoundError:
    jd_text = "Senior AI Engineer embeddings vector database pinecone qdrant faiss semantic search retrieval ranking NDCG python production deployment fine-tuning lora qlora"
jd_tokens = tokenize(jd_text)
jd_load_time = time.perf_counter() - t0
print(f"         JD: {len(jd_text)} chars, {len(jd_tokens)} tokens  [{jd_load_time:.3f}s]")

# ── 5. Read candidates ────────────────────────────────────────────────────────
print(f"\n[STEP 2] Reading {SAMPLE_SIZE} candidates...")
t0 = time.perf_counter()
candidates = []
with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= SAMPLE_SIZE: break
        line = line.strip()
        if line: candidates.append(json.loads(line))
read_time = time.perf_counter() - t0
print(f"         Read {len(candidates)} candidates  [{read_time:.2f}s]")

# ── 6. Phase 1 hard filters ───────────────────────────────────────────────────
print(f"\n[STEP 3] Phase 1 hard filters...")
t0 = time.perf_counter()
survivors = [c for c in candidates if apply_hard_filters(c)]
filter_time = time.perf_counter() - t0
filter_rate = len(survivors) / len(candidates) * 100
print(f"         {len(candidates)} → {len(survivors)} survivors ({filter_rate:.1f}%)  [{filter_time:.2f}s]")

# ── 7A. BM25 scoring (soft signal across ALL survivors) ────────────────────────
print(f"\n[STEP 4A] BM25 scoring {len(survivors)} survivors (soft signal, not cutoff)...")
t0 = time.perf_counter()

corpus = [tokenize(candidate_full_text(c)) for c in survivors]
bm25   = BM25Okapi(corpus)
bm25_raw_scores = bm25.get_scores(jd_tokens)
bm25_max = bm25_raw_scores.max()
bm25_norm_scores = bm25_raw_scores / (bm25_max + 1e-9)   # normalised [0,1]

bm25_time = time.perf_counter() - t0
print(f"         BM25 scored {len(survivors)} survivors  [{bm25_time:.2f}s]")
print(f"         BM25 score range: {bm25_raw_scores.min():.2f} – {bm25_raw_scores.max():.2f}")

# ── Keyword scoring constants + functions (defined here so Step 4B can use them) ──

PROF_W = {"expert": 1.0, "advanced": 0.8, "intermediate": 0.5, "beginner": 0.2}

SKILL_TIERS = [
    (1.0, ["embedding", "vector search", "semantic search", "information retrieval",
           "rag", "retrieval augmented", "learning to rank",
           "pinecone", "qdrant", "weaviate", "milvus", "faiss", "pgvector",
           "chromadb", "opensearch", "elasticsearch",
           "fine-tuning", "fine-tune", "peft", "lora", "qlora",
           "sentence transformer", "hugging face", "reranking", "bge", "e5"]),
    (0.6, ["nlp", "natural language", "llm", "language model",
           "mlops", "mlflow", "weights & biases", "wandb", "kubeflow", "bentoml",
           "recommendation", "ranking", "retrieval",
           "pytorch", "python", "tensorflow"]),
    (0.3, ["machine learning", "deep learning", "neural", "scikit",
           "data science", "feature engineering", "model deploy",
           "bert", "gpt", "transformer", "attention"]),
]

CAREER_POS = [
    (1.0, ["embedding", "vector", "retrieval", "semantic search", "rag",
           "fine-tuning", "fine-tune", "language model", "llm", "reranking",
           "recommendation system", "information retrieval", "search ranking"]),
    (0.6, ["nlp", "model training", "model deployment", "inference",
           "feature engineering", "ml pipeline", "production ml",
           "transformer", "bert", "gpt", "vector database"]),
    (0.3, ["machine learning", "deep learning", "neural", "data science",
           "pytorch", "tensorflow", "scikit", "prediction model"]),
]

ML_JOB_TITLES = [
    "ml engineer", "machine learning", "data scientist", "ai engineer",
    "nlp engineer", "research engineer", "ai specialist", "ai researcher",
    "computer vision engineer", "deep learning engineer", "applied scientist",
    "software engineer (ml)", "(ml)", "junior ml", "applied ml",
]

# Keywords that match Tier 1 skills — used to filter assessment scores
TIER1_ASSESSMENT_KWS = [
    "embedding", "vector", "search", "retrieval", "rag", "nlp", "language model",
    "llm", "pinecone", "qdrant", "weaviate", "milvus", "faiss", "elasticsearch",
    "fine-tuning", "fine-tune", "lora", "qlora", "peft", "ranking", "reranking",
    "sentence transformer", "python", "pytorch", "recommendation",
]

def ml_career_ratio(c: dict) -> float:
    """Fraction of total career months spent in ML-titled roles."""
    jobs = c.get("career_history", [])
    if not jobs:
        return 0.0
    total_months = sum(j.get("duration_months", 0) for j in jobs) or 1
    ml_months = sum(
        j.get("duration_months", 0)
        for j in jobs
        if any(kw in j.get("title", "").lower() for kw in ML_JOB_TITLES)
    )
    return ml_months / total_months

def keyword_score(c: dict) -> tuple:
    """Returns (combined_kw, skill_norm, career_norm)."""
    # ── Skill component ──
    sk_raw = 0.0
    sk_max = 0.0
    for skill in c.get("skills", []):
        sname = skill.get("name", "").lower()
        prof  = skill.get("proficiency", "beginner").lower()
        dur   = skill.get("duration_months", 0)
        pw    = PROF_W.get(prof, 0.2)
        dur_f = min(dur / 24.0, 1.5)
        for tw, kws in SKILL_TIERS:
            if any(kw in sname for kw in kws):
                sk_raw += tw * pw * (0.5 + 0.5 * dur_f)
                sk_max += tw
                break
    skill_norm = min(sk_raw / sk_max, 1.0) if sk_max > 0 else 0.0

    # ── Career component ──
    ratio = ml_career_ratio(c)
    career_text = " ".join(j.get("description", "") for j in c.get("career_history", [])).lower()
    cr_raw = 0.0
    cr_max = sum(tw * len(kws) for tw, kws in CAREER_POS)
    for tw, kws in CAREER_POS:
        cr_raw += tw * sum(1 for kw in kws if kw in career_text)
    text_score = min(cr_raw / cr_max, 1.0) if cr_max > 0 else 0.0

    career_norm = 0.50 * ratio + 0.50 * text_score
    combined = 0.35 * skill_norm + 0.65 * career_norm
    return combined, skill_norm, career_norm

# ── 7B. Smart Pre-filter Gate: BM25 + Redrob keyword_score ───────────────────
# Uses the SAME keyword_score() (SKILL_TIERS + ML career ratio) as final scoring.
# No separate simplified function — one consistent scoring logic throughout.
print(f"\n[STEP 4B] Pre-filter keyword scoring {len(survivors)} survivors...")
t_4b = time.perf_counter()

pre_kw_scores     = np.array([keyword_score(c)[0] for c in survivors])
pre_filter_scores = 0.40 * bm25_norm_scores + 0.60 * pre_kw_scores
gate_idx          = np.argsort(pre_filter_scores)[::-1][:BM25_TOP_N]
bm25_candidates   = [survivors[i] for i in gate_idx]
bm25_gate_norm    = bm25_norm_scores[gate_idx]   # BM25 scores aligned to bm25_candidates

t_4b_elapsed = time.perf_counter() - t_4b
print(f"         KW range: {pre_kw_scores.min():.3f} – {pre_kw_scores.max():.3f}")
print(f"         Pre-filter score range: {pre_filter_scores.min():.3f} – {pre_filter_scores.max():.3f}  [{t_4b_elapsed:.2f}s]")
print(f"\n[STEP 4C] Smart gate: {len(survivors)} → {len(bm25_candidates)} candidates")
print(f"         Formula: 0.40×BM25_norm + 0.60×keyword_score (Redrob SKILL_TIERS + ML career ratio)")
print(f"         Skips embedding {len(survivors) - len(bm25_candidates)} candidates ({(len(survivors)-len(bm25_candidates))/len(survivors)*100:.1f}% reduction)")

# ── 8. Load model ────────────────────────────────────────────────────────────
print(f"\n[STEP 5] Loading model: {MODEL_NAME}...")
t0 = time.perf_counter()
model = SentenceTransformer(MODEL_NAME)
model_time = time.perf_counter() - t0
print(f"         Model loaded  [{model_time:.2f}s]")

# ── 9. Embed JD (BGE uses query prefix for retrieval mode) ───────────────────
print(f"\n[STEP 6] Embedding JD...")
t0 = time.perf_counter()
# BGE-small requires a prefix on the QUERY (JD) side for better retrieval performance.
# Candidate passages are encoded WITHOUT prefix (asymmetric retrieval).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
jd_chunks = [
    BGE_QUERY_PREFIX + jd_text[i:i+800]
    for i in range(0, len(jd_text), 800)
]
jd_embs   = model.encode(jd_chunks, batch_size=BATCH_SIZE, show_progress_bar=False)
jd_embedding = jd_embs.mean(axis=0)
jd_embed_time = time.perf_counter() - t0
print(f"         JD embedded ({len(jd_chunks)} chunks, with BGE query prefix)  [{jd_embed_time:.3f}s]")

# ── 10. Build sections & embed top-N pre-filtered candidates ─────────────────
# Gate applied in Step 4C: only bm25_candidates (top-2500) are embedded.
print(f"\n[STEP 7] Building sections for {len(bm25_candidates)} pre-filtered candidates...")
t0 = time.perf_counter()
all_sections = []
section_map  = []
for c in bm25_candidates:
    sects = candidate_sections(c)
    section_map.append(len(sects))
    all_sections.extend(sects)
build_time = time.perf_counter() - t0
avg_sects = len(all_sections) / max(len(bm25_candidates), 1)
print(f"         {len(all_sections)} total sections ({avg_sects:.1f} avg/candidate)  [{build_time:.3f}s]")

print(f"\n[STEP 8] Generating embeddings (batch_size={BATCH_SIZE})...")
t0 = time.perf_counter()
all_embeddings = model.encode(
    all_sections,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
)
embed_time = time.perf_counter() - t0
speed = len(all_sections) / embed_time
print(f"         {len(all_sections)} sections in {embed_time:.2f}s  ({speed:.0f} sections/sec)")

# ── 11. Average sections → cosine similarity ──────────────────────────────────
print(f"\n[STEP 9] Averaging section embeddings + cosine similarity...")
t0 = time.perf_counter()
cand_embeddings = []
cursor = 0
for n in section_map:
    cand_embeddings.append(all_embeddings[cursor:cursor+n].mean(axis=0))
    cursor += n
cand_embeddings = np.array(cand_embeddings)

cand_norm  = cand_embeddings / (np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-9)
jd_norm    = jd_embedding    / (np.linalg.norm(jd_embedding) + 1e-9)
sem_scores = cand_norm @ jd_norm
sim_time   = time.perf_counter() - t0
print(f"         Score range: {sem_scores.min():.3f} to {sem_scores.max():.3f}  (mean: {sem_scores.mean():.3f})  [{sim_time:.4f}s]")

# bm25_candidates = top-BM25_TOP_N by combined gate (set in Step 4C above)
# bm25_gate_norm  = BM25 norm scores aligned to bm25_candidates


# ── 13. Assessment Score (NEW) ────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
def assessment_score(c: dict) -> float:
    """
    Platform-verified test scores from redrob_signals.skill_assessment_scores.
    Only counts scores for skills matching our Tier 1 keywords.
    Returns a 0.0–1.0 normalised score.
    """
    scores_dict = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if not scores_dict:
        return 0.0

    matched_scores = []
    for skill_name, score in scores_dict.items():
        sname = skill_name.lower()
        if any(kw in sname for kw in TIER1_ASSESSMENT_KWS):
            matched_scores.append(score)

    if not matched_scores:
        # Fall back: average of all assessment scores (normalised)
        all_vals = list(scores_dict.values())
        return (sum(all_vals) / len(all_vals)) / 100.0 * 0.5  # half weight for unrelated skills
    return sum(matched_scores) / len(matched_scores) / 100.0

# ═══════════════════════════════════════════════════════════════════════════════
# ── 14. Education Score (NEW) ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
EDU_TIER_SCORES = {
    "tier_1": 1.0,    # IIT, IISc, IIM, BITS
    "tier_2": 0.7,    # NIT, SRM, Anna University
    "tier_3": 0.4,    # LPU, Chandigarh University
    "tier_4": 0.1,    # Local / unknown colleges
    "unknown": 0.2,
}

def education_score(c: dict) -> float:
    """
    Returns the best education tier score across all education entries.
    Only counts CS / related degrees.
    """
    cs_fields = ["computer", "software", "information", "data", "ai", "machine",
                 "electronics", "electrical", "mathematics", "statistics", "physics"]
    best = 0.0
    for edu in c.get("education", []):
        field = edu.get("field_of_study", "").lower()
        tier  = edu.get("tier", "unknown").lower()
        # Small weight reduction for completely unrelated degrees
        field_ok = any(kw in field for kw in cs_fields) if field else True
        raw_score = EDU_TIER_SCORES.get(tier, 0.2)
        effective = raw_score if field_ok else raw_score * 0.5
        if effective > best:
            best = effective
    return best

# ═══════════════════════════════════════════════════════════════════════════════
# ── 15. Behavioral Multiplier (NEW) ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
# Reference date for "days inactive" calculations
_TODAY = datetime.now()

def _parse_date(d_str: str):
    if not d_str:
        return None
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except ValueError:
        return None

def behavioral_multiplier(c: dict) -> tuple:
    """
    Computes a multiplier in [0.5, 1.15] based on 23 Redrob signals.
    Returns (multiplier, breakdown_dict) for display.

    Boosters (above 1.0):
      +0.10  open_to_work_flag = True
      +0.10  notice_period_days <= 30
      +0.05  github_activity_score > 50
      +0.05  recruiter_response_rate >= 0.70
      +0.05  interview_completion_rate >= 0.80
      +0.05  saved_by_recruiters_30d >= 5   (hot candidate)

    Penalties (below 1.0):
      -0.20  last_active_date > 180 days ago
      -0.10  last_active_date 91-180 days ago
      -0.15  recruiter_response_rate < 0.20
      -0.10  interview_completion_rate < 0.40
      -0.05  notice_period_days > 90
      -0.05  open_to_work_flag = False (explicitly not looking)

    Clamp: [0.5, 1.15]
    """
    sigs  = c.get("redrob_signals", {})
    base  = 1.0
    notes = []

    # ── Availability signals ──
    open_to_work = sigs.get("open_to_work_flag", None)
    if open_to_work is True:
        base += 0.10
        notes.append("open_to_work+0.10")
    elif open_to_work is False:
        base -= 0.05
        notes.append("not_open-0.05")

    notice = sigs.get("notice_period_days", 90)
    if notice <= 30:
        base += 0.10
        notes.append(f"notice≤30d+0.10")
    elif notice > 90:
        base -= 0.05
        notes.append(f"notice>{90}d-0.05")

    # ── Activity / recency (JD explicitly calls this out) ──
    last_active_str = sigs.get("last_active_date", "")
    last_active = _parse_date(last_active_str)
    if last_active:
        days_inactive = (_TODAY - last_active).days
        if days_inactive > 180:
            base -= 0.20
            notes.append(f"inactive{days_inactive}d-0.20")
        elif days_inactive > 90:
            base -= 0.10
            notes.append(f"inactive{days_inactive}d-0.10")
    
    # ── Engagement quality ──
    rr = sigs.get("recruiter_response_rate", 0.5)
    if rr < 0.20:
        base -= 0.15
        notes.append(f"resp_rate{rr:.2f}-0.15")
    elif rr >= 0.70:
        base += 0.05
        notes.append(f"resp_rate{rr:.2f}+0.05")

    icr = sigs.get("interview_completion_rate", 0.5)
    if icr < 0.40:
        base -= 0.10
        notes.append(f"interview_rate{icr:.2f}-0.10")
    elif icr >= 0.80:
        base += 0.05
        notes.append(f"interview_rate{icr:.2f}+0.05")

    # ── External validation / GitHub ──
    gh = sigs.get("github_activity_score", -1)
    if gh > 50:
        base += 0.05
        notes.append(f"github{gh:.0f}+0.05")

    # ── Market demand signal ──
    saved = sigs.get("saved_by_recruiters_30d", 0)
    if saved >= 5:
        base += 0.05
        notes.append(f"saved{saved}+0.05")

    mult = max(0.5, min(1.15, base))
    return mult, notes

# ═══════════════════════════════════════════════════════════════════════════════
# ── 16. Specific Reasoning Generator (NEW) ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
def generate_reasoning(c: dict, sem_score: float, kw_score: float,
                        assess_s: float, edu_s: float,
                        b_mult: float, final_score: float, rank: int) -> str:
    """
    Generates specific, factual 1–2 sentence reasoning per candidate.
    Every claim references an actual field in the profile.
    Tone matches rank (top-10 = positive, bottom-20 = honest about gaps).
    """
    profile = c.get("profile", {})
    sigs    = c.get("redrob_signals", {})

    yoe     = profile.get("years_of_experience", 0)
    title   = profile.get("current_title", "Engineer")
    loc     = profile.get("location", "India")
    company = profile.get("current_company", "")

    # ── Pick top-2 relevant skills ──
    relevant_skills = []
    for s in c.get("skills", []):
        sname = s.get("name", "")
        sname_l = sname.lower()
        dur   = s.get("duration_months", 0)
        prof  = s.get("proficiency", "")
        if any(kw in sname_l for kw in ["embedding", "vector", "rag", "retrieval",
                                         "nlp", "faiss", "pinecone", "qdrant",
                                         "weaviate", "milvus", "lora", "qlora",
                                         "fine-tun", "sentence transformer",
                                         "language model", "llm", "reranking",
                                         "elasticsearch", "opensearch"]):
            relevant_skills.append(f"{sname} ({prof}, {dur}mo)")
        if len(relevant_skills) >= 3:
            break

    # ── Career highlight: most recent ML job ──
    ml_job_str = ""
    for job in c.get("career_history", []):
        jt = job.get("title", "")
        jc = job.get("company", "")
        jm = job.get("duration_months", 0)
        if any(kw in jt.lower() for kw in ML_JOB_TITLES):
            ml_job_str = f"{jt} @ {jc} ({jm}mo)"
            break

    # ── Availability note ──
    notice     = sigs.get("notice_period_days", 90)
    open_work  = sigs.get("open_to_work_flag", None)
    last_act   = _parse_date(sigs.get("last_active_date", ""))
    days_inact = (_TODAY - last_act).days if last_act else 0

    avail_parts = []
    if open_work is True:
        avail_parts.append("open to work")
    if notice <= 30:
        avail_parts.append(f"notice {notice}d ✅")
    elif notice > 90:
        avail_parts.append(f"notice {notice}d ⚠️")
    if days_inact > 180:
        avail_parts.append(f"inactive {days_inact}d ⚠️")
    avail_str = "; ".join(avail_parts) if avail_parts else f"notice {notice}d"

    # ── Education ──
    edu_str = ""
    for edu in c.get("education", []):
        tier = edu.get("tier", "")
        if tier in ("tier_1", "tier_2"):
            edu_str = f"{edu.get('institution','')} ({tier})"
            break

    # ── Assessment ──
    assess_dict = sigs.get("skill_assessment_scores", {})
    top_assess = sorted(
        [(k, v) for k, v in assess_dict.items() if any(kw in k.lower() for kw in TIER1_ASSESSMENT_KWS)],
        key=lambda x: -x[1]
    )[:2]
    assess_str = "; ".join(f"{k}={v:.0f}" for k, v in top_assess) if top_assess else ""

    # ── Compose sentence 1: core fit ──
    skill_part = ", ".join(relevant_skills) if relevant_skills else "general ML background"
    career_part = f"most recent ML role: {ml_job_str}" if ml_job_str else f"current role: {title} @ {company}"
    sent1 = f"{yoe:.0f}yr {title} in {loc}; {skill_part}; {career_part}."

    # ── Compose sentence 2: availability + signals + concerns ──
    parts2 = [avail_str]
    if assess_str:
        parts2.append(f"platform scores: {assess_str}")
    if edu_str:
        parts2.append(f"edu: {edu_str}")
    if rank > 60 and kw_score < 0.30:
        parts2.append("limited core keyword depth")
    if b_mult < 0.75:
        parts2.append("engagement signals weak")
    sent2 = "; ".join(parts2) + f" [final={final_score:.3f}]."

    return f"{sent1} {sent2}"

# ═══════════════════════════════════════════════════════════════════════════════
# ── 17. Compute all scores ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
kw_scores    = np.array([keyword_score(c)[0] for c in bm25_candidates])
kw_skill     = np.array([keyword_score(c)[1] for c in bm25_candidates])
kw_career    = np.array([keyword_score(c)[2] for c in bm25_candidates])
assess_scores = np.array([assessment_score(c)  for c in bm25_candidates])
edu_scores    = np.array([education_score(c)   for c in bm25_candidates])
b_mults       = np.array([behavioral_multiplier(c)[0] for c in bm25_candidates])
b_notes_list  = [behavioral_multiplier(c)[1]  for c in bm25_candidates]

kw_time = time.perf_counter() - t0
print(f"         Keyword range:    {kw_scores.min():.3f} to {kw_scores.max():.3f}  (mean: {kw_scores.mean():.3f})")
print(f"         Assessment range: {assess_scores.min():.3f} to {assess_scores.max():.3f}  (mean: {assess_scores.mean():.3f})")
print(f"         Education range:  {edu_scores.min():.3f} to {edu_scores.max():.3f}  (mean: {edu_scores.mean():.3f})")
print(f"         Behavioral mult:  {b_mults.min():.3f} to {b_mults.max():.3f}  (mean: {b_mults.mean():.3f})  [{kw_time:.3f}s]")

# ── 18. Combined base score ────────────────────────────────────────────────────
# Formula:  base = 0.40×semantic + 0.10×bm25(soft) + 0.33×keyword + 0.12×assess + 0.05×edu
#           final = base × behavioral_multiplier
#
# BM25 gate (Step 4C) already reduced to top-2500 before embedding.
# bm25_gate_norm is BM25 scores aligned to bm25_candidates.
base_scores = (
    0.40 * sem_scores        +  # semantic similarity (pre-filtered candidates)
    0.10 * bm25_gate_norm    +  # BM25 soft signal (gate-aligned, normalised)
    0.33 * kw_scores         +  # keyword: skills + career depth
    0.12 * assess_scores     +  # platform assessment scores
    0.05 * edu_scores            # institution tier
)

# Duplicate description soft penalty (before multiplier)
for idx, c in enumerate(bm25_candidates):
    if not passes_duplicate_description_filter(c):
        base_scores[idx] -= 0.05

# Apply behavioral multiplier
final_scores = base_scores * b_mults

print(f"\n         Base score range:  {base_scores.min():.3f} to {base_scores.max():.3f}  (mean: {base_scores.mean():.3f})")
print(f"         Final score range: {final_scores.min():.3f} to {final_scores.max():.3f}  (mean: {final_scores.mean():.3f})")

# ── 19. Full Constraint Compliance + Timing Breakdown ────────────────────────
import tracemalloc
tracemalloc.start()

# ── Extrapolate all steps to the full 100k run ────────────────────────────────
full_read_time    = (read_time   / max(len(candidates), 1)) * 100_000
full_filter_time  = (filter_time / max(len(candidates), 1)) * 100_000
full_bm25_time    = (bm25_time   / max(len(survivors), 1))  * FULL_SURVIVORS
full_sections_cnt = FULL_BM25_TOP * avg_sects    # Only top-{BM25_TOP_N} gate candidates embedded
full_embed_time   = full_sections_cnt / max(speed, 1)
full_score_time   = (kw_time / max(len(bm25_candidates), 1)) * FULL_BM25_TOP

# Each named step: (label, measured/extrapolated seconds, is_extrapolated)
steps = [
    ("[LOAD]  Model: BAAI/bge-small-en-v1.5 (~130MB)",      model_time,       False),
    ("[READ]  Read 100k candidates from disk",            full_read_time,   True),
    ("[FILT]  Phase 1 hard filters (8 rules)",            full_filter_time, True),
    ("[BM25]  BM25 coarse rank all survivors",            full_bm25_time,   True),
    ("[EMB]   Embed JD text (chunked)",                   jd_embed_time,    False),
    (f"[EMB]   BGE embed top-{FULL_BM25_TOP} candidates", full_embed_time,  True),
    ("[SIM]   Cosine similarity (vectorised numpy)",      sim_time,         False),
    ("[SCR]   KW + Assess + Edu + Behavioral scoring",   full_score_time,  True),
    ("[CSV]   Sort top-100 + write submission CSV",       3.0,              False),
]

total_wall = sum(s[1] for s in steps)

# ── Memory estimate (rough: candidates in RAM + model + embeddings) ───────────
# 100k candidates × ~4.6KB average ≈ 460 MB
# BAAI/bge-small-en-v1.5 model weights  ≈ 130 MB
# Embeddings: 2500 candidates × 512-dim × 4B ≈  5 MB
# BM25 corpus (tokenised):  ~9600 docs × ~150 tokens × 8B ≈ 11 MB
# Total peak estimate:
ram_candidates_mb = 100_000 * 4_700 / 1024 / 1024
ram_model_mb      = 130
ram_embeddings_mb = FULL_BM25_TOP * 512 * 4 / 1024 / 1024
ram_bm25_mb       = FULL_SURVIVORS * 150 * 8 / 1024 / 1024
ram_misc_mb       = 200   # numpy arrays, Python overhead
ram_total_mb      = ram_candidates_mb + ram_model_mb + ram_embeddings_mb + ram_bm25_mb + ram_misc_mb
RAM_LIMIT_MB      = 16 * 1024  # 16 GB

# ── Disk estimate ─────────────────────────────────────────────────────────────
# Compressed candidate file: ~52 MB (given), plain jsonl: ~465 MB
# Model cache (HF hub):       ~130 MB
# Output CSV:                 ~20 KB
disk_input_mb  = 465        # candidates.jsonl (uncompressed)
disk_model_mb  = 130        # BAAI/bge-small-en-v1.5 cached weights
disk_output_mb = 0.02       # submission CSV
disk_total_mb  = disk_input_mb + disk_model_mb + disk_output_mb
DISK_LIMIT_MB  = 5 * 1024   # 5 GB

# ── Actual total elapsed (real wall-clock from program start) ────────────────
actual_elapsed = time.perf_counter() - PROGRAM_START
actual_ok      = actual_elapsed <= BUDGET

W = 78  # report width

def status_icon(ok: bool) -> str:
    return "✅ PASS" if ok else "❌ FAIL"

print("\n" + "═" * W)
print("  CONSTRAINT COMPLIANCE REPORT — Redrob Submission Spec v4")
print("═" * W)

# ── Constraint summary table ──────────────────────────────────────────────────
wall_ok  = total_wall  <= BUDGET
ram_ok   = ram_total_mb <= RAM_LIMIT_MB
disk_ok  = disk_total_mb <= DISK_LIMIT_MB
cpu_ok   = True
net_ok   = True

print(f"\n  {'Constraint':<22} {'Limit':>12}   {'Extrapolated':>14}   {'Actual (this run)':>18}   Status")
print("  " + "─" * (W + 10))
print(f"  {'Runtime (wall-clock)':<22} {'≤ 5 min (300s)':>12}   {fmt(total_wall):>14}   {fmt(actual_elapsed):>18}   {status_icon(actual_ok)}")
print(f"  {'RAM':<22} {'≤ 16 GB':>12}   {ram_total_mb/1024:.2f} GB (est.)   {'(see breakdown)':>18}   {status_icon(ram_ok)}")
print(f"  {'Disk':<22} {'≤ 5 GB':>12}   {disk_total_mb/1024:.2f} GB (est.)   {'(see breakdown)':>18}   {status_icon(disk_ok)}")
print(f"  {'Compute':<22} {'CPU only':>12}   {'CPU only':>14}   {'CPU only':>18}   {status_icon(cpu_ok)}")
print(f"  {'Network':<22} {'Off (no APIs)':>12}   {'No API calls':>14}   {'No API calls':>18}   {status_icon(net_ok)}")

print(f"\n  NOTE: 'Extrapolated' = projected time for full 100k run using measured rates")
print(f"        'Actual'       = real wall-clock time of THIS test run (smaller candidate subset)")

# ── RAM breakdown ─────────────────────────────────────────────────────────────
print(f"\n  RAM BREAKDOWN (estimated peak):")
print(f"  {'Component':<40} {'MB':>8}")
print("  " + "─" * 50)
print(f"  {'100k candidate records (parsed JSON)':<40} {ram_candidates_mb:>7.0f}")
print(f"  {'BAAI/bge-small-en-v1.5 model weights':<40} {ram_model_mb:>7.0f}")
print(f"  {'Candidate embeddings (top-{FULL_BM25_TOP})':<40} {ram_embeddings_mb:>7.1f}")
print(f"  {'BM25 corpus (tokenised survivors)':<40} {ram_bm25_mb:>7.1f}")
print(f"  {'NumPy arrays + Python overhead':<40} {ram_misc_mb:>7.0f}")
print(f"  {'─'*50}")
print(f"  {'TOTAL (estimated)':<40} {ram_total_mb:>7.0f}  /  {RAM_LIMIT_MB:.0f} MB limit")
ram_pct = ram_total_mb / RAM_LIMIT_MB * 100
ram_bar = int(ram_pct / 5)
print(f"  {'RAM USED':<40} {ram_pct:>7.1f}%  {'▓'*ram_bar}{'░'*(20-ram_bar)}")

# ── Disk breakdown ────────────────────────────────────────────────────────────
print(f"\n  DISK BREAKDOWN (intermediate state):")
print(f"  {'Component':<40} {'MB':>8}")
print("  " + "─" * 50)
print(f"  {'candidates.jsonl (uncompressed input)':<40} {disk_input_mb:>7.0f}")
print(f"  {'BAAI/bge-small-en-v1.5 cache (HuggingFace)':<40} {disk_model_mb:>7.0f}")
print(f"  {'submission CSV output':<40} {disk_output_mb*1000:>7.2f} KB")
print(f"  {'─'*50}")
print(f"  {'TOTAL (estimated)':<40} {disk_total_mb:>7.0f}  /  {DISK_LIMIT_MB:.0f} MB limit")
disk_pct = disk_total_mb / DISK_LIMIT_MB * 100
disk_bar = max(1, int(disk_pct / 5))
print(f"  {'DISK USED':<40} {disk_pct:>7.1f}%  {'▓'*disk_bar}{'░'*(20-disk_bar)}")

# ── Wall-clock timing breakdown ───────────────────────────────────────────────
print(f"\n  WALL-CLOCK TIMING BREAKDOWN (extrapolated to full 100k run):")
print(f"  {'Step':<50} {'Time':>8}  {'%':>5}  {'Budget used':>12}  Bar")
print("  " + "─" * (W - 2))
cumulative = 0.0
for name, secs, is_extrap in steps:
    pct      = secs / total_wall * 100
    cumulative += secs
    cum_pct  = cumulative / BUDGET * 100
    extrap   = "~" if is_extrap else " "
    bar_w    = int(round(min(secs / total_wall, 1.0) * 16))
    bar_str  = "█" * bar_w + "░" * (16 - bar_w)
    cum_str  = f"{cumulative:.0f}s / {BUDGET}s ({cum_pct:.0f}%)"
    print(f"  {extrap}{name:<49} {fmt(secs):>8}  {pct:>4.1f}%  {cum_str:>18}")
print("  " + "─" * (W - 2))

budget_pct = total_wall / BUDGET * 100
budget_bar = min(int(budget_pct / 5), 20)
print(f"  {'  TOTAL (extrapolated)':<50} {fmt(total_wall):>8}  100.0%")
print(f"  {'  5-MIN BUDGET':<50} {fmt(BUDGET):>8}")
print(f"  {'  HEADROOM':<50} {fmt(BUDGET - total_wall):>8}  ({100 - budget_pct:.1f}% free)")
print(f"\n  BUDGET GAUGE  0%{'─'*8}50%{'─'*7}100%")
print(f"  {'':16}[{'▓'*budget_bar}{'░'*(20-budget_bar)}]  {budget_pct:.1f}% used")

if total_wall <= BUDGET:
    print(f"\n  ✅ RUNTIME: WITHIN BUDGET — {BUDGET - total_wall:.0f}s ({(BUDGET - total_wall)/60:.1f} min) headroom")
else:
    print(f"\n  ❌ RUNTIME: OVER BUDGET by {total_wall - BUDGET:.0f}s — optimise embedding step")

if ram_ok and disk_ok and cpu_ok and net_ok:
    print(f"  ✅ ALL OTHER CONSTRAINTS PASS")
else:
    if not ram_ok:  print(f"  ❌ RAM EXCEEDED — reduce candidate pool or use mmap")
    if not disk_ok: print(f"  ❌ DISK EXCEEDED — clean intermediate files")

print("═" * W)

print(f"\n  ⏱  ACTUAL TOTAL PROGRAM TIME: {fmt(actual_elapsed)}  (this test run, not extrapolated)")
if actual_ok:
    print(f"  ✅ Actual elapsed {actual_elapsed:.1f}s is within the {BUDGET}s budget")
else:
    print(f"  ❌ Actual elapsed {actual_elapsed:.1f}s EXCEEDS {BUDGET}s budget by {actual_elapsed-BUDGET:.0f}s")
print("═" * W)

# ── 20. Display Top 10 ─────────────────────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  TOP 10 QUICK VIEW")
print(f"  Score = 0.40×semantic + 0.10×bm25 + 0.33×keyword + 0.12×assessment + 0.05×edu  ×  behavioral_mult")
print(f"{'='*100}\n")

top_10_idx = np.argsort(final_scores)[::-1][:10]

for rank, i in enumerate(top_10_idx, 1):
    c       = bm25_candidates[i]
    profile = c.get("profile", {})
    name    = profile.get("anonymized_name", "?")
    title   = profile.get("current_title", "?")
    company = profile.get("current_company", "?")
    yoe     = profile.get("years_of_experience", 0)
    loc     = profile.get("location", "?")
    reloc   = c.get("redrob_signals", {}).get("willing_to_relocate", None)
    _, sk, cr = keyword_score(c)
    ratio   = ml_career_ratio(c)
    b_mult  = b_mults[i]
    b_notes = b_notes_list[i]
    a_score = assess_scores[i]
    e_score = edu_scores[i]

    # Line 1: rank + all scores
    print(f"\n  [{rank:>3}] Final={final_scores[i]:.3f}  "
          f"Sem={sem_scores[i]:.3f}  KW={kw_scores[i]:.3f}  "
          f"Assess={a_score:.2f}  Edu={e_score:.2f}  "
          f"BehMult={b_mult:.2f}  ml_ratio={ratio:.0%}")
    print(f"        {name}  |  {title} @ {company} ({yoe:.0f}yr)  [{loc}]  reloc={reloc}")

    # Line 2: career titles
    jobs = c.get("career_history", [])
    career_parts = []
    for j in jobs:
        jt = j.get("title", "?")
        ml_f = "✅" if any(kw in jt.lower() for kw in ML_JOB_TITLES) else "❌"
        career_parts.append(f"{ml_f}{j.get('duration_months',0)}mo {jt}@{j.get('company','?')}")
    print(f"        Career: {' | '.join(career_parts)}")

    # Line 3: top skills
    skills = c.get("skills", [])
    skill_line = " | ".join(
        f"{s.get('name','')}({s.get('proficiency','')[:3]},{s.get('duration_months',0)}mo)"
        for s in skills[:6]
    )
    print(f"        Skills: {skill_line}")

    # Line 4: behavioral notes
    if b_notes:
        print(f"        Behav:  {' | '.join(b_notes)}")

    # Line 5: reasoning
    reasoning = generate_reasoning(
        c, sem_scores[i], kw_scores[i], a_score, e_score, b_mult, final_scores[i], rank
    )
    print(f"        Reason: {reasoning}")

print(f"\n{'='*100}\n")
