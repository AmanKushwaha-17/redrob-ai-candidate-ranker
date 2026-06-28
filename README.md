# Redrob AI Candidate Ranking Pipeline

## Pipeline Overview

```
100,000 candidates
     │
     ▼ Stage 1 — Hard Filters (8 rules)          ~9,600 survivors
     │
     ▼ Stage 2 — BM25 Scoring (all survivors)    ~1.5s
     │
     ▼ Stage 3 — Smart Gate                      9,600 → 2,000
     │           BM25_norm×0.40 + keyword_score×0.60
     │
     ▼ Stage 4 — BGE-small Embedding             ~207s
     │           BAAI/bge-small-en-v1.5 (512-dim, retrieval-tuned)
     │
     ▼ Stage 5 — Final Scoring (5 signals)       final × behavioral_mult
     │
     ▼  Top 100 → team_submission.csv
```

---

## Stage 1: Hard Filters — *Why*: Eliminate structurally unqualifiable candidates instantly

These are deterministic O(1) checks that run on all 100k candidates in ~10s. Every filter maps to an explicit disqualifier stated or implied in the JD.

| Filter | Drop Reason |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Honeypot** | Profiles with impossible data (e.g. 0 months tenure + Expert proficiency). Synthetic noise injected to catch copy-paste submissions. |
| **Location** | JD requires India-based or relocation-willing candidates. Pure remote-only or overseas profiles are out. |
| **Consulting** | Candidates who spent 100% of their career at service firms (TCS, Infosys, Wipro, etc.) with no product-company experience. These engineers rarely own end-to-end ML systems. |
| **Closed Source** | 5+ year engineers with zero external validation (no GitHub, papers, or talks). JD explicitly values open-source contribution. |
| **Title Chaser** | Average tenure ≤ 1.5 years with inflated Senior/Principal titles. Pattern indicates resume padding, not skill depth. |
| **Architecture Astronaut** | Senior engineers who haven't shipped production code in 18+ months. JD needs hands-on builders, not architects. |
| **Domain Expertise** | CV/Speech/Robotics specialists with no NLP or IR exposure. Domain mismatch for an embeddings/retrieval role. |
| **Title Mismatch** | Profiles with AI keywords but non-engineering titles (Marketing, HR, Finance). Common false-positive trap. |

---

## Stage 2: BM25 Scoring — *Why*: Fast exact-match signal across all 9,600 survivors

BM25 (Okapi BM25) is a classical term-frequency retrieval model. It scores each candidate's full text (summary + job descriptions + skills) against JD tokens in ~1.5s.

**Why BM25 and not just keyword matching?**
BM25 applies TF-IDF weighting with length normalisation. Candidates who mention "vector search" 10 times in a long document don't score unfairly over someone who mentions it 3 times in a tight summary. It also penalises extremely long documents that dilute term density.

**BM25 is a soft signal only** — it does not act as a cutoff gate alone. Its normalised score [0,1] feeds into Stage 3.

---

## Stage 3: Smart Gate — *Why*: Reduce to 2,000 before the expensive embedding step

Embedding 9,600 candidates with BGE-small takes ~1,200s — 4× over the 5-minute budget. We must reduce first.

**Gate formula:**

```
gate_score = 0.40 × BM25_norm + 0.60 × keyword_score
```

**Why combine both?**

- BM25 alone misses candidates who use synonyms or describe skills implicitly
- keyword_score alone ignores whether the JD terms appear frequently or contextually
- Together they catch exact-match depth (BM25) AND domain skill proficiency (keyword)

**keyword_score** is a weighted function of:

- **Skill tiers** (Tier 1 = vector DBs, fine-tuning, RAG; Tier 2 = NLP/LLM tools; Tier 3 = general ML) × proficiency level × tenure duration
- **ML career ratio** — fraction of total career months spent in ML-titled roles × career description keyword hits

Only the **top 2,000 by gate_score** proceed to embedding. This saves ~51s of embedding time vs top-2500, giving ~68s headroom within the 300s budget.

---

## Stage 4: BGE-small Embedding — *Why*: Semantic understanding beyond keyword matching

Model: `BAAI/bge-small-en-v1.5` (512-dim, ~130MB, CPU-only, ~207s for 2000 candidates)

**Why BGE-small over all-MiniLM-L6-v2?**

| | MiniLM-L6 | BGE-small |
| --------------- | ------------ | ----------------------------------- |
| Sem score range | 0.107–0.586 | 0.621–0.840 |
| Trained for | General NLU | Retrieval / search (BEIR, MS-MARCO) |
| Query prefix | No | Yes — asymmetric encoding |
| Dim | 384 | 512 |

BGE's asymmetric encoding uses a query-side prefix for the JD and plain passage encoding for candidates. This is how it was trained — using it correctly gives a ~25% better semantic separation on retrieval tasks.

**Section averaging:** Each candidate is split into 3–4 sections (summary + job descriptions). Each section is embedded independently, then averaged. This avoids the 512-token limit truncating key content from long profiles.

---

## Stage 5: Final Scoring — *Why*: No single signal is sufficient alone

```
base  = 0.40×semantic + 0.10×BM25_norm + 0.33×keyword + 0.12×assessment + 0.05×education
final = base × behavioral_multiplier  (clamped [0.5, 1.15])
```

| Signal | Weight | Reason |
| ----------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Semantic** (BGE cosine sim) | 40% | Captures intent match beyond keywords — finds candidates who describe vector search without saying "vector search" |
| **Keyword score** (tiered) | 33% | Domain depth signal: Tier 1 retrieval skills weighted higher than general ML. Proficiency + tenure prevent padding. |
| **Assessment score** | 12% | Platform-verified test results. Only Tier 1 relevant skill assessments count; unrelated scores get 50% weight. |
| **Education** | 5% | Tier 1 institutions (IIT, IISc, BITS, MIT) signal baseline rigor. Non-CS degrees get 50% reduction. |
| **BM25 norm** | 10% | Keeps exact JD-term density as a small tiebreaker signal. |
| **Behavioral multiplier** | ×[0.5–1.15] | Boosts: open-to-work, notice ≤30d, GitHub activity, recruiter engagement. Penalises: inactive >180d, low response rate, long notice. |

---

## Constraint Compliance

| Constraint | Limit | Actual |
| ---------- | -------- | ----------------------- |
| Runtime | ≤ 300s | ~232s ✅ (68s headroom) |
| RAM | ≤ 16 GB | ~794 MB ✅ |
| Disk | ≤ 5 GB | ~595 MB ✅ |
| Compute | CPU only | CPU only ✅ |
| Network | Off | No API calls ✅ |

---

## Files

| File | Purpose |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| `ranker.py` | **Production entry point.** Run this to generate `team_submission.csv`. |
| `filters.py` | All 8 hard-filter functions (Stage 1). |
| `download_model.py` | **Pre-computation step.** Downloads BGE model weights locally before offline ranking. |
| `job_description.md` | Job description used for JD embedding in the ranking pipeline. |
| `test_semantic.py` | Full benchmark with constraint report, timing breakdown, and top-10 display. For testing only. |
| `view_final_100.py` | Utility to inspect the final ranked output. |
| `team_submission.csv` | Output: top-100 candidates with rank, score, reasoning. |
| `models/bge-small-en-v1.5/` | Local model cache — config and tokenizer files only. Weights downloaded via download_model.py. |

---

## How to Reproduce

### Step 1 — Install dependencies
```bash
pip install rank_bm25 sentence-transformers torch numpy
```

### Step 2 — Download model (pre-computation, needs network, run once)
```bash
python download_model.py
```
Downloads `BAAI/bge-small-en-v1.5` (~130 MB) from HuggingFace and saves it to
`models/bge-small-en-v1.5/`. After this step, no network access is needed.

### Step 3 — Run the ranker (no network, CPU only, ≤ 300s)
```bash
python ranker.py
```

Output: `team_submission.csv` — 100 ranked candidates with scores and reasoning.

> **Stage 3 reproduction note:** Steps 1 and 2 are pre-computation and may be run before
> the sandboxed timing window begins. Step 3 (ranker.py) is the ranking step — it runs
> fully offline, CPU-only, and completes in ~232s on a 12-core machine.
