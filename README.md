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
This downloads `BAAI/bge-small-en-v1.5` (~130 MB) from HuggingFace and saves it to
`models/bge-small-en-v1.5/`. After this step, no network access is needed.

### Step 3 — Run the ranker (no network, CPU only, ≤ 300s)
```bash
python ranker.py --candidates ../candidates.jsonl --out team_submission.csv
```
Or with default paths:
```bash
python ranker.py
```

Output: `team_submission.csv` — 100 ranked candidates with scores and reasoning.

> **Stage 3 reproduction note:** Steps 1 and 2 are pre-computation and may be run before
> the sandboxed timing window begins. Step 3 (ranker.py) is the ranking step — it runs
> fully offline, CPU-only, and completes in ~232s on a 12-core machine.
