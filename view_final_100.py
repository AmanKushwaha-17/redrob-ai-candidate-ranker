import json
import csv
import os
import gzip
import sys

# Config
CANDIDATES_FILE = r"c:\Users\amank\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
CSV_FILE = r"c:\Users\amank\Downloads\[PUB] India_runs_data_and_ai_challenge\ranker_project\team_submission.csv"

# Load candidate IDs and scores/ranks from CSV
top_100 = {}
try:
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            top_100[row["candidate_id"]] = {
                "rank": row["rank"],
                "score": row["score"],
                "reasoning": row["reasoning"]
            }
except FileNotFoundError:
    print(f"Could not find {CSV_FILE}")
    exit(1)

# Extract just the top 100 candidates from the 100k file
found_candidates = {}
opener = gzip.open if CANDIDATES_FILE.endswith(".gz") else open
with opener(CANDIDATES_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        c = json.loads(line)
        cid = c.get("candidate_id")
        if cid in top_100:
            found_candidates[cid] = c
            if len(found_candidates) == len(top_100):
                break

print("\n==========================================================================================")
print("  FINAL TOP 100 FULL PROFILE VIEW")
print("==========================================================================================\n")

# Sort them by rank
ranked_cids = sorted(top_100.keys(), key=lambda x: int(top_100[x]["rank"]))

ML_JOB_TITLES = [
    "ml engineer", "machine learning", "data scientist", "ai engineer",
    "nlp engineer", "research engineer", "ai specialist", "ai researcher",
    "computer vision engineer", "deep learning engineer", "applied scientist",
    "software engineer (ml)", "(ml)", "junior ml", "applied ml",
]

for cid in ranked_cids:
    c = found_candidates.get(cid)
    if not c: continue
    
    meta = top_100[cid]
    rank = int(meta["rank"])
    score = meta["score"]
    
    profile = c.get("profile", {})
    name    = profile.get("anonymized_name", "?")
    title   = profile.get("current_title", "?")
    comp    = profile.get("current_company", "?")
    yoe     = profile.get("years_of_experience", 0)
    loc     = profile.get("location", "?")
    reloc   = profile.get("willing_to_relocate", False)

    print(f"  [{rank:2d}] COMB={score}  |  {cid}")
    print(f"       {name}  |  {title} @ {comp} ({yoe}yr)  [{loc}]  reloc={reloc}")
    
    # Career summary
    jobs = c.get("career_history", [])
    career_str = []
    for j in jobs:
        jt = j.get("title", "")
        jm = j.get("duration_months", 0)
        jc = j.get("company", "")
        # mark if ML or not
        ml_f = "✅" if any(kw in jt.lower() for kw in ML_JOB_TITLES) else "❌"
        career_str.append(f"{ml_f}{jm}mo {jt}@{jc}")
    print(f"       Career: {' | '.join(career_str)}")
    
    # Skill summary (top 6)
    skills = c.get("skills", [])
    sk_str = []
    for s in skills[:6]:
        sk_str.append(f"{s.get('name')}({s.get('proficiency')[:3]})")
    print(f"       Skills: {' | '.join(sk_str)}")
    
    print(f"       Reasoning: {meta['reasoning']}")
    print()
