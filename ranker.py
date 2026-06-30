import os, sys, json, time, re, gzip, csv, argparse
import numpy as np

# ── Dependencies ──────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    sys.exit("[ERROR] pip install rank_bm25")

try:
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
except ImportError:
    sys.exit("[ERROR] pip install sentence-transformers")

from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filters import (
    is_honeypot, passes_location_filter, passes_consulting_filter,
    passes_closed_source_filter, passes_title_chaser_filter,
    passes_architecture_astronaut_filter, passes_domain_expertise_filter,
    passes_title_mismatch_filter, passes_duplicate_description_filter,
)

# ── Config ────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "models", "bge-small-en-v1.5")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
BM25_TOP_N = 2000
BATCH_SIZE = 128
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Helpers ───────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

def candidate_full_text(c):
    p = c.get("profile", {})
    parts = [p.get("summary",""), p.get("headline",""), p.get("current_title","")]
    for j in c.get("career_history", []):
        parts += [j.get("title",""), j.get("description","")]
    for s in c.get("skills", []):
        parts.append(s.get("name",""))
    return " ".join(x for x in parts if x)

def candidate_sections(c):
    sects = []
    s = c.get("profile", {}).get("summary", "").strip()
    if s: sects.append(s)
    for j in c.get("career_history", []):
        d = j.get("description", "").strip()
        if d: sects.append(d)
    return sects or [c.get("profile", {}).get("headline", "")]

def apply_hard_filters(c):
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

# ── Keyword Scoring ───────────────────────────────────────────────────────────
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
           "recommendation", "ranking", "retrieval", "pytorch", "python", "tensorflow"]),
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

TIER1_ASSESSMENT_KWS = [
    "embedding", "vector", "search", "retrieval", "rag", "nlp", "language model",
    "llm", "pinecone", "qdrant", "weaviate", "milvus", "faiss", "elasticsearch",
    "fine-tuning", "fine-tune", "lora", "qlora", "peft", "ranking", "reranking",
    "sentence transformer", "python", "pytorch", "recommendation",
]

EDU_TIER_SCORES = {"tier_1": 1.0, "tier_2": 0.7, "tier_3": 0.4, "tier_4": 0.1, "unknown": 0.2}

CS_FIELDS = ["computer", "software", "information", "data", "ai", "machine",
             "electronics", "electrical", "mathematics", "statistics", "physics"]

_TODAY = datetime.now()

def ml_career_ratio(c):
    jobs = c.get("career_history", [])
    if not jobs: return 0.0
    total = sum(j.get("duration_months", 0) for j in jobs) or 1
    ml = sum(j.get("duration_months", 0) for j in jobs
             if any(kw in j.get("title","").lower() for kw in ML_JOB_TITLES))
    return ml / total

def keyword_score(c):
    sk_raw = sk_max = 0.0
    for skill in c.get("skills", []):
        sname = skill.get("name", "").lower()
        pw    = PROF_W.get(skill.get("proficiency","beginner").lower(), 0.2)
        dur_f = min(skill.get("duration_months", 0) / 24.0, 1.5)
        for tw, kws in SKILL_TIERS:
            if any(kw in sname for kw in kws):
                sk_raw += tw * pw * (0.5 + 0.5 * dur_f)
                sk_max += tw
                break
    skill_norm = min(sk_raw / sk_max, 1.0) if sk_max > 0 else 0.0

    ratio = ml_career_ratio(c)
    career_text = " ".join(j.get("description","") for j in c.get("career_history",[])).lower()
    cr_max = sum(tw * len(kws) for tw, kws in CAREER_POS)
    cr_raw = sum(tw * sum(1 for kw in kws if kw in career_text) for tw, kws in CAREER_POS)
    text_score = min(cr_raw / cr_max, 1.0) if cr_max > 0 else 0.0

    career_norm = 0.50 * ratio + 0.50 * text_score
    return 0.35 * skill_norm + 0.65 * career_norm, skill_norm, career_norm

def assessment_score(c):
    scores = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if not scores: return 0.0
    matched = [v for k, v in scores.items()
               if any(kw in k.lower() for kw in TIER1_ASSESSMENT_KWS)]
    if matched:
        return sum(matched) / len(matched) / 100.0
    all_v = list(scores.values())
    return (sum(all_v) / len(all_v)) / 100.0 * 0.5

def education_score(c):
    best = 0.0
    for edu in c.get("education", []):
        field = edu.get("field_of_study", "").lower()
        field_ok = any(kw in field for kw in CS_FIELDS) if field else True
        raw = EDU_TIER_SCORES.get(edu.get("tier","unknown").lower(), 0.2)
        eff = raw if field_ok else raw * 0.5
        best = max(best, eff)
    return best

def _parse_date(s):
    try: return datetime.strptime(s, "%Y-%m-%d") if s else None
    except ValueError: return None

def behavioral_multiplier(c):
    sigs = c.get("redrob_signals", {})
    base = 1.0
    if sigs.get("open_to_work_flag") is True:  base += 0.10
    elif sigs.get("open_to_work_flag") is False: base -= 0.05
    notice = sigs.get("notice_period_days", 90)
    if notice <= 30:  base += 0.10
    elif notice > 90: base -= 0.05
    last = _parse_date(sigs.get("last_active_date",""))
    if last:
        days = (_TODAY - last).days
        if days > 180: base -= 0.20
        elif days > 90: base -= 0.10
    rr = sigs.get("recruiter_response_rate", 0.5)
    if rr < 0.20: base -= 0.15
    elif rr >= 0.70: base += 0.05
    icr = sigs.get("interview_completion_rate", 0.5)
    if icr < 0.40: base -= 0.10
    elif icr >= 0.80: base += 0.05
    if sigs.get("github_activity_score", -1) > 50: base += 0.05
    if sigs.get("saved_by_recruiters_30d", 0) >= 5:  base += 0.05
    return max(0.5, min(1.15, base))

def generate_reasoning(c, sem, kw, assess, edu, b_mult, final, rank):
    p    = c.get("profile", {})
    sigs = c.get("redrob_signals", {})
    yoe  = p.get("years_of_experience", 0)
    title = p.get("current_title", "Engineer")
    loc   = p.get("location", "India")

    top_skills = []
    for s in c.get("skills", []):
        sn = s.get("name","")
        if any(kw in sn.lower() for t in SKILL_TIERS[:2] for kw in t[1]):
            top_skills.append(f"{sn} ({s.get('proficiency','')}, {s.get('duration_months',0)}mo)")
        if len(top_skills) == 3: break

    recent_job = next(iter(c.get("career_history",[])), {})
    recent_str = f"{recent_job.get('title','')} @ {recent_job.get('company','')} ({recent_job.get('duration_months',0)}mo)" if recent_job else ""

    scores_dict = c.get("redrob_signals",{}).get("skill_assessment_scores",{})
    matched_scores = {k: v for k, v in scores_dict.items()
                     if any(kw in k.lower() for kw in TIER1_ASSESSMENT_KWS)}
    assess_str = "; ".join(f"{k}={v}" for k,v in list(matched_scores.items())[:2]) if matched_scores else ""

    edu_list = c.get("education", [])
    edu_str = f"{edu_list[0].get('institution','')} ({edu_list[0].get('tier','')})" if edu_list else ""

    notice = sigs.get("notice_period_days", 90)
    if notice <= 30:
        notice_str = f"notice {notice}d (immediate join)"
    elif notice > 90:
        notice_str = f"notice {notice}d (long notice, potential concern)"
    else:
        notice_str = f"notice {notice}d"
    open_str = "open to work" if sigs.get("open_to_work_flag") else "not actively looking"

    parts = [f"{yoe:.0f}yr {title} in {loc}"]
    if top_skills: parts.append("; ".join(top_skills))
    if recent_str: parts.append(f"most recent: {recent_str}")
    if open_str:   parts.append(open_str)
    if notice_str: parts.append(notice_str)
    if assess_str: parts.append(f"platform scores: {assess_str}")
    if edu_str:    parts.append(f"edu: {edu_str}")
    parts.append(f"[final={final:.3f}]")
    return ". ".join(parts)

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    # ── Argument parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Redrob AI Candidate Ranker — Team Lanzers")
    parser.add_argument(
        "--candidates",
        default=os.environ.get("CANDIDATES_FILE", os.path.join(_DIR, "candidates.jsonl")),
        help="Path to candidates.jsonl or candidates.jsonl.gz"
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("OUTPUT_CSV", os.path.join(_DIR, "Lanzers.csv")),
        help="Output CSV path (default: ./Lanzers.csv)"
    )
    parser.add_argument(
        "--jd",
        default=os.path.join(_DIR, "job_description.md"),
        help="Path to job description markdown file (default: ./job_description.md)"
    )
    args = parser.parse_args()

    CANDIDATES_FILE = args.candidates
    OUTPUT_CSV      = args.out
    JD_FILE         = args.jd

    t0_total = time.perf_counter()
    print("=" * 60)
    print("  REDROB CANDIDATE RANKER — Team Lanzers")
    print("=" * 60)
    print(f"  Candidates : {CANDIDATES_FILE}")
    print(f"  Output     : {OUTPUT_CSV}")
    print(f"  JD         : {JD_FILE}")
    print("=" * 60)

    # 1. Load JD
    print("[1/7] Loading Job Description...")
    try:
        with open(JD_FILE, "r", encoding="utf-8") as f:
            jd_text = f.read()
    except FileNotFoundError:
        print(f"      WARNING: {JD_FILE} not found — using built-in fallback JD text.")
        jd_text = ("Senior AI Engineer embeddings vector database pinecone qdrant faiss "
                   "semantic search retrieval ranking NDCG python production fine-tuning lora qlora "
                   "sentence transformers BGE information retrieval hybrid search reranking "
                   "mlops evaluation framework learning to rank product company")
    jd_tokens = tokenize(jd_text)

    # 2. Read & hard-filter candidates
    print("[2/7] Reading and hard-filtering candidates...")
    survivors = []
    opener = gzip.open if CANDIDATES_FILE.endswith(".gz") else open
    with opener(CANDIDATES_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            c = json.loads(line)
            if apply_hard_filters(c):
                survivors.append(c)
    print(f"      Survivors: {len(survivors)}")

    # 3. BM25 score all survivors
    print("[3/7] BM25 scoring survivors...")
    corpus    = [tokenize(candidate_full_text(c)) for c in survivors]
    bm25      = BM25Okapi(corpus)
    bm25_raw  = bm25.get_scores(jd_tokens)
    bm25_norm = bm25_raw / (bm25_raw.max() + 1e-9)

    # 4. Smart gate: BM25 + keyword_score → top-2000
    print("[4/7] Smart gate (BM25 + keyword_score) → top-2000...")
    kw_gate     = np.array([keyword_score(c)[0] for c in survivors])
    gate_scores = 0.40 * bm25_norm + 0.60 * kw_gate
    gate_idx    = np.argsort(gate_scores)[::-1][:BM25_TOP_N]
    candidates  = [survivors[i] for i in gate_idx]
    bm25_gate   = bm25_norm[gate_idx]
    print(f"      {len(survivors)} → {len(candidates)} candidates")

    # 5. Load BGE-small model (local cache → HuggingFace fallback)
    print("[5/7] Loading BAAI/bge-small-en-v1.5...")
    model_src    = MODEL_PATH if os.path.isdir(MODEL_PATH) else MODEL_NAME
    model        = SentenceTransformer(model_src)
    jd_chunks    = [BGE_QUERY_PREFIX + jd_text[i:i+800] for i in range(0, len(jd_text), 800)]
    jd_embedding = model.encode(jd_chunks, batch_size=BATCH_SIZE, show_progress_bar=False).mean(axis=0)

    # 6. Embed candidates (passage side — no prefix)
    print("[6/7] Embedding 2000 candidates...")
    all_sections, section_map = [], []
    for c in candidates:
        sects = candidate_sections(c)
        section_map.append(len(sects))
        all_sections.extend(sects)

    all_embs = model.encode(all_sections, batch_size=BATCH_SIZE,
                            show_progress_bar=True, convert_to_numpy=True)
    cand_embs, cursor = [], 0
    for n in section_map:
        cand_embs.append(all_embs[cursor:cursor+n].mean(axis=0))
        cursor += n
    cand_embs = np.array(cand_embs)

    cand_norm  = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-9)
    jd_norm    = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-9)
    sem_scores = cand_norm @ jd_norm

    # 7. Final scoring + write CSV
    print("[7/7] Scoring and writing submission...")
    results = []
    for i, c in enumerate(candidates):
        kw, _, _ = keyword_score(c)
        assess   = assessment_score(c)
        edu      = education_score(c)
        b_mult   = behavioral_multiplier(c)
        dup_pen  = -0.05 if not passes_duplicate_description_filter(c) else 0.0
        base     = 0.40 * sem_scores[i] + 0.10 * bm25_gate[i] + 0.33 * kw + 0.12 * assess + 0.05 * edu
        final    = max(0.0, base * b_mult + dup_pen)
        results.append({
            "c": c, "final": final,
            "sem": sem_scores[i], "kw": kw,
            "assess": assess, "edu": edu, "b_mult": b_mult,
        })

    results.sort(key=lambda x: x["final"], reverse=True)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(results[:100], 1):
            c      = r["c"]
            cid    = c.get("candidate_id", "")
            reason = generate_reasoning(
                c, r["sem"], r["kw"], r["assess"], r["edu"], r["b_mult"], r["final"], rank
            )
            writer.writerow([cid, rank, f"{r['final']:.5f}", reason])
            if rank <= 10:
                p = c.get("profile", {})
                print(f"  [{rank:2d}] {p.get('anonymized_name','?')} | {p.get('current_title','?')} "
                      f"| final={r['final']:.3f} sem={r['sem']:.3f} kw={r['kw']:.3f}")

    elapsed = time.perf_counter() - t0_total
    print(f"\n✅ Done in {elapsed:.1f}s  |  Submission → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
