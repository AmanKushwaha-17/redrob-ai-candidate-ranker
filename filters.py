from datetime import datetime

def parse_date(d_str):
    """Safely parse a YYYY-MM-DD date string."""
    if not d_str:
        return datetime.now()
    try:
        return datetime.strptime(d_str, "%Y-%m-%d")
    except ValueError:
        return datetime.now()

def is_honeypot(candidate: dict) -> bool:
    """
    Identifies honeypot candidates based on subtly impossible profiles.
    Returns True if the candidate is deemed a honeypot, False otherwise.
    """
    
    # 1. Skill mismatch: Expert/Advanced proficiency with 0 months of use
    # Also catches candidates with many skills listed at 0 duration.
    zero_duration_count = 0
    for skill in candidate.get("skills", []):
        prof = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)
        
        if prof in ["expert", "advanced"] and duration == 0:
            return True
            
        if duration == 0:
            zero_duration_count += 1
            
    # "expert proficiency in 10 skills with 0 years used"
    if zero_duration_count >= 10:
        return True

    # 2. Career duration mismatch: duration_months is heavily inflated compared to dates
    total_career_months = 0
    for job in candidate.get("career_history", []):
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        
        # End date before start date
        if end < start:
            return True
            
        diff_months = (end.year - start.year) * 12 + (end.month - start.month)
        duration = job.get("duration_months", 0)
        
        # Stated duration is significantly longer than the actual calendar difference
        # Giving a generous 3-month leeway for rounding
        if duration > diff_months + 3:
            return True
            
        total_career_months += duration

    # 3. Overall years of experience inflated
    # Profile says they have X years of experience, but their career history
    # accounts for vastly less (e.g., claiming 10 years when they only have 2 years of history)
    # We give a 5-year (60 month) leeway because some people omit early jobs
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if yoe * 12 > total_career_months + 60:
        return True

    # If no impossible traits found
    return False

def passes_location_filter(candidate: dict) -> bool:
    """
    Returns True if the candidate passes the hard location filter.
    JD allows: Pune, Noida, Hyderabad, Mumbai, Delhi NCR + major tech hubs.
    Rules:
      - willing_to_relocate = True  → Pass regardless of location
      - willing_to_relocate = False OR None (unspecified) + overseas → Fail
      - In India + valid city → Pass
      - In India + unknown city + not willing → Fail
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    
    # Only pass on explicit True — None/False/missing all treated as unwilling
    willing = signals.get("willing_to_relocate", None)
    
    country = profile.get("country", "").lower()
    location = profile.get("location", "").lower()

    # Detect if candidate is overseas (country set AND not India)
    is_overseas = bool(country) and country != "india" and "india" not in location

    if is_overseas:
        # Only pass if EXPLICITLY willing to relocate (True), not None or False
        return willing is True

    # Candidate is in India (or country unknown → assume India)
    # If explicitly willing → pass
    if willing is True:
        return True

    # Check if they are in a valid city
    valid_cities = ["pune", "noida", "hyderabad", "mumbai", "delhi", "ncr",
                    "bangalore", "bengaluru", "gurgaon", "gurugram", "chennai",
                    "kolkata", "ahmedabad", "kochi", "coimbatore"]
    
    for city in valid_cities:
        if city in location:
            return True
            
    # In India, not in a valid city, and not explicitly willing to relocate
    return False


def passes_consulting_filter(candidate: dict) -> bool:
    """
    JD Disqualifier: "People who have only worked at consulting firms 
    (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) in their entire career."
    If they have at least one job outside this list, they pass.
    """
    consulting_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
    
    career_history = candidate.get("career_history", [])
    if not career_history:
        return True
        
    for job in career_history:
        company = job.get("company", "").lower()
        
        is_consulting = False
        for firm in consulting_firms:
            if firm in company:
                is_consulting = True
                break
                
        if not is_consulting:
            # Found at least one job that is NOT a consulting firm
            return True
            
    # If we get here, EVERY job was at a consulting firm
    return False

def passes_closed_source_filter(candidate: dict) -> bool:
    """
    JD Disqualifier: "People whose work has been entirely on closed-source 
    proprietary systems for 5+ years without external validation (papers, talks, open-source)."
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    
    # If they have less than 5 years of experience, this rule doesn't disqualify them
    if yoe < 5:
        return True
        
    # Check for open source activity on GitHub
    github_score = candidate.get("redrob_signals", {}).get("github_activity_score", -1)
    if github_score > 0:
        return True
        
    # Check for keywords related to papers, talks, open-source in profile summary and descriptions
    keywords = ["paper", "talk", "speaker", "conference", "open-source", "open source", "publication", "published", "arxiv", "patent"]
    
    text_to_search = candidate.get("profile", {}).get("summary", "").lower()
    for job in candidate.get("career_history", []):
        text_to_search += " " + job.get("description", "").lower()
        
    for kw in keywords:
        if kw in text_to_search:
            return True
            
    # If they have 5+ years experience, no GitHub activity, and no mention of external validation
    return False

def passes_title_chaser_filter(candidate: dict) -> bool:
    """
    JD Disqualifier: "Title-chasers. If your career trajectory shows you optimizing for 
    'Senior' -> 'Staff' -> 'Principal' titles by switching companies every 1.5 years..."
    """
    career = candidate.get("career_history", [])
    
    # We group durations by company so that promotions within the same company 
    # aren't penalized as "switching companies"
    companies = {}
    for job in career:
        comp = job.get("company", "").strip().lower()
        if not comp: continue
        companies[comp] = companies.get(comp, 0) + job.get("duration_months", 0)
        
    if len(companies) < 3:
        # Can't be a serial hopper if they've only worked at 1 or 2 companies
        return True
        
    avg_company_duration = sum(companies.values()) / len(companies)
    
    if avg_company_duration <= 18:
        # Their average tenure at a company is 1.5 years or less.
        # Check if they hold a senior/inflated title currently.
        current_title = candidate.get("profile", {}).get("current_title", "").lower()
        senior_titles = ["senior", "staff", "principal", "lead", "director", "vp", "head", "manager", "chief", "architect"]
        
        for t in senior_titles:
            if t in current_title:
                return False
                
    return True


def passes_architecture_astronaut_filter(candidate: dict) -> bool:
    """
    JD Disqualifier: "Senior engineer who hasn't written production code in the last 18 months 
    because you've moved into 'architecture' or 'tech lead' roles."
    """
    career = candidate.get("career_history", [])
    if not career:
        return True
        
    current_job = career[0]
    for job in career:
        if job.get("is_current", False):
            current_job = job
            break
            
    title = current_job.get("title", "").lower()
    duration = current_job.get("duration_months", 0)
    
    # If they are purely an Architect or Tech Lead (and not holding an "Engineer" title)
    if ("architect" in title or "tech lead" in title) and "engineer" not in title:
        # And they've been in this non-coding role for > 18 months
        if duration > 18:
            desc = current_job.get("description", "").lower()
            # If their job description contains zero mention of actually writing code
            coding_words = ["code", "coding", "programmed", "programming", "developed", "built", "hands-on", "implemented"]
            if not any(cw in desc for cw in coding_words):
                return False
                
    return True

def passes_domain_expertise_filter(candidate: dict) -> bool:
    """
    JD Disqualifier: "People whose primary expertise is computer vision, speech, 
    or robotics without significant NLP/IR exposure."

    Checks TWO signals:
      1. Skills: CV skill months vs NLP skill months
      2. Career titles: did they actually WORK in CV/NLP roles?

    A candidate can inject NLP skills but if their actual job titles are all
    'Computer Vision Engineer' / 'CV Researcher' with no NLP career role,
    they are still a CV specialist (e.g. Aarav Goyal pattern).
    """
    cv_skill_kws  = ["computer vision", "image processing", "vision", "object detection",
                     "yolo", "opencv", "speech", "audio", "robotics", "ros"]
    nlp_skill_kws = ["nlp", "natural language", "llm", "language model",
                     "information retrieval", "search", "text", "transformer",
                     "bert", "gpt", "rag", "embedding", "retrieval"]

    cv_job_kws  = ["computer vision", "cv engineer", "vision engineer",
                   "image recognition", "robotics engineer"]
    nlp_job_kws = ["nlp", "natural language", "language model", "search engineer",
                   "information retrieval", "ml engineer", "machine learning",
                   "data scientist", "ai engineer", "ai specialist", "ai researcher",
                   "applied scientist", "research engineer"]

    # ── Skill signal ──
    cv_months  = 0
    nlp_months = 0
    for s in candidate.get("skills", []):
        name     = s.get("name", "").lower()
        duration = s.get("duration_months", 0)
        if any(kw in name for kw in cv_skill_kws):
            cv_months  += duration
        if any(kw in name for kw in nlp_skill_kws):
            nlp_months += duration

    # If skills are purely CV with zero NLP → drop
    if cv_months > 0 and nlp_months == 0:
        return False

    # ── Career title signal ──
    # If they have NLP skills listed BUT every actual job title is a CV role
    # with NO NLP career experience → they are a CV specialist with injected skills
    if nlp_months > 0:
        jobs = candidate.get("career_history", [])
        if jobs:
            cv_career_months  = sum(
                j.get("duration_months", 0) for j in jobs
                if any(kw in j.get("title", "").lower() for kw in cv_job_kws)
            )
            nlp_career_months = sum(
                j.get("duration_months", 0) for j in jobs
                if any(kw in j.get("title", "").lower() for kw in nlp_job_kws)
            )
            # They have CV career experience but zero NLP career experience
            # — the NLP skills are aspirational/injected, not real
            if cv_career_months > 0 and nlp_career_months == 0:
                return False

    return True


def passes_duplicate_description_filter(candidate: dict) -> bool:
    """
    Drops candidates whose career history contains two or more roles
    with word-for-word identical descriptions.

    In real profiles, every job has a unique description. Identical descriptions
    across different companies/roles is a clear synthetic data artifact
    (e.g., Zara Malhotra: Data Scientist @ Glance and ML Engineer @ Zomato
    had the EXACT same job description — impossible in reality).
    """
    descriptions = []
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").strip().lower()
        if desc:
            descriptions.append(desc)

    # If any two descriptions are identical → synthetic/fake profile
    seen = set()
    for d in descriptions:
        if d in seen:
            return False
        seen.add(d)

    return True


def passes_title_mismatch_filter(candidate: dict) -> bool:
    """
    Hackathon Trap: "A candidate who has all the AI keywords listed as skills 
    but whose title is 'Marketing Manager' is not a fit..."
    Also drops non-ML domain titles like Civil/Mechanical Engineer.
    """
    current_title = candidate.get("profile", {}).get("current_title", "").lower()
    if not current_title:
        return True
        
    trap_titles = [
        # Business / non-technical roles
        "marketing", "sales", "hr", "human resources", "recruiter",
        "accountant", "finance", "customer support", "customer service",
        "operations manager", "teacher", "trainer", "educator",
        "content writer", "content creator", "seo", "copywriter",
        "graphic designer", "ux designer", "ui designer", "visual designer",
        "project manager", "program manager", "product owner",
        "business analyst", "business development",
        "legal", "lawyer", "compliance", "auditor",
        # Non-CS engineering
        "civil engineer", "mechanical engineer", "electrical engineer",
        "structural engineer", "chemical engineer", "aerospace engineer",
        "industrial engineer", "manufacturing engineer", "process engineer",
        "hardware engineer", "embedded engineer",
        # Other domains
        "doctor", "physician", "nurse", "pharmacist",
        "architect",          # building architect — not software
        "supply chain", "logistics", "procurement",
    ]
    
    for t in trap_titles:
        if t in current_title:
            return False
                
    return True

if __name__ == "__main__":
    # Quick local test logic to verify it works
    import json
    import os
    import time
    
    # Optional test against candidates.jsonl if it exists in the parent directory
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    candidates_file = os.path.join(parent_dir, "[PUB] India_runs_data_and_ai_challenge", "India_runs_data_and_ai_challenge", "candidates.jsonl")
    
    if os.path.exists(candidates_file):
        count = 0
        honeypot_count = 0
        location_fail_count = 0
        consulting_fail_count = 0
        closed_source_fail_count = 0
        title_chaser_fail_count = 0
        architecture_astronaut_fail_count = 0
        domain_expertise_fail_count = 0
        title_mismatch_fail_count = 0
        duplicate_desc_fail_count = 0

        print(f"Starting filter tests...")
        time_start = time.time()
        
        with open(candidates_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                c = json.loads(line)
                count += 1
                
                if is_honeypot(c):
                    honeypot_count += 1
                    continue
                    
                if not passes_location_filter(c):
                    location_fail_count += 1
                    continue
                    
                if not passes_consulting_filter(c):
                    consulting_fail_count += 1
                    continue
                    
                if not passes_closed_source_filter(c):
                    closed_source_fail_count += 1
                    continue
                    
                if not passes_title_chaser_filter(c):
                    title_chaser_fail_count += 1
                    continue
                    
                if not passes_architecture_astronaut_filter(c):
                    architecture_astronaut_fail_count += 1
                    continue
                    
                if not passes_domain_expertise_filter(c):
                    domain_expertise_fail_count += 1
                    continue

                if not passes_title_mismatch_filter(c):
                    title_mismatch_fail_count += 1
                    continue
                    

        time_end = time.time()
        elapsed = time_end - time_start
        
        print(f"Total processed: {count}")
        print(f"Honeypots detected (dropped): {honeypot_count}")
        print(f"Failed location filter (dropped): {location_fail_count}")
        print(f"Failed consulting filter (dropped): {consulting_fail_count}")
        print(f"Failed closed-source filter (dropped): {closed_source_fail_count}")
        print(f"Failed title-chaser filter (dropped): {title_chaser_fail_count}")
        print(f"Failed architecture astronaut filter (dropped): {architecture_astronaut_fail_count}")
        print(f"Failed domain expertise filter (dropped): {domain_expertise_fail_count}")
        print(f"Failed title mismatch filter (dropped): {title_mismatch_fail_count}")
        # Note: passes_duplicate_description_filter exists but is NOT applied as a hard filter
        # — catches 2290 candidates which exceeds the ~100 honeypot budget (synthetic data artifact)

        total_dropped = (honeypot_count + location_fail_count + consulting_fail_count +
                         closed_source_fail_count + title_chaser_fail_count +
                         architecture_astronaut_fail_count + domain_expertise_fail_count +
                         title_mismatch_fail_count)
        print(f"Candidates remaining: {count - total_dropped}")
        print(f"Time taken: {elapsed:.4f} seconds")
        if elapsed > 0:
            print(f"Speed: {count / elapsed:.0f} candidates/second")

