#!/usr/bin/env python3
"""
Literature Pipeline — Query arXiv, filter, summarize, compare with Agentic WM.
Usage: python3 literature/pipeline.py [--days 7] [--max 20]
"""

import argparse, json, os, re, sys, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

# ── Config ──────────────────────────────────────────────
LIT_DIR = os.path.dirname(os.path.abspath(__file__))
INBOX_DIR = os.path.join(LIT_DIR, "inbox")

QUERIES = [
    # World Model + Robotics
    'all:"world model" AND (all:robot OR all:manipulation OR all:embodiment)',
    # Video Foundation Model + Action
    '(all:"video prediction" OR all:"video foundation model") AND (all:action OR all:control OR all:robot)',
    # Agent / Tool Use + Embodiment
    '(all:"tool use" OR all:"function calling" OR all:tool-use) AND (all:robot OR all:"world model")',
    # Pretrain + Fine-tune + Video + Robot
    'all:pretrain AND all:video AND (all:robot OR all:manipulation)',
    # LLM path / agentic world model (exact match check)
    'all:"agentic world model" OR all:"world model agent"',
]

# Agentic WM Framework properties (for comparison)
OUR_FRAMEWORK = {
    "frame_as_single_token": True,    # One vector per frame
    "decoupled_action": True,          # Action not in base model
    "tool_use_interface": True,        # Action = tool call
    "two_layer_prediction": True,      # Natural + Intervention
    "pretrain_then_finetune": True,    # Stage 1 → Stage 2
    "llm_paradigm_replication": True,  # Explicit meta-claim
    "persistent_state": True,          # Global state token
    "causal_transformer": True,        # Causal attention
}


# ── arXiv API ───────────────────────────────────────────
def query_arxiv(query: str, max_results: int = 50, days_back: int = 14) -> list[dict]:
    """Query arXiv API, return list of paper dicts."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": min(max_results, 100),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = base_url + "?" + urllib.parse.urlencode(params)
    
    req = urllib.request.Request(url, headers={"User-Agent": "AgenticWM-LitPipeline/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode("utf-8")
            break
        except Exception as e:
            if attempt < 2:
                wait = (attempt + 1) * 10
                print(f"  [RETRY] Attempt {attempt+1} failed ({e}), waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [WARN] Query failed after 3 attempts: {e}")
                return []
    
    root = ET.fromstring(data)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    papers = []
    
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.find("atom:id", ns).text.strip().split("/abs/")[-1]
        title = " ".join(entry.find("atom:title", ns).text.strip().split())
        summary = " ".join(entry.find("atom:summary", ns).text.strip().split())
        published = entry.find("atom:published", ns).text.strip()
        pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
        
        if pub_date < cutoff:
            continue
        
        authors = [
            a.find("atom:name", ns).text.strip()
            for a in entry.findall("atom:author", ns)
        ]
        
        # Categories
        cats = [c.get("term") for c in entry.findall("atom:category", ns)]
        
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "summary": summary[:1500],
            "authors": authors,
            "published": published,
            "categories": cats,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "query": query[:60],
        })
    
    return papers


# ── Dedup ───────────────────────────────────────────────
def deduplicate(papers: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for p in papers:
        if p["arxiv_id"] not in seen:
            seen.add(p["arxiv_id"])
            unique.append(p)
    return unique


# ── Relevance scoring ──────────────────────────────────
RELEVANCE_KEYWORDS = {
    "world model": 3,
    "video prediction": 2,
    "next-frame": 3,
    "tool use": 4,
    "function call": 4,
    "agentic": 5,
    "decoupled": 4,
    "pretrain": 2,
    "foundation model": 2,
    "robot manipulation": 3,
    "embodied": 2,
    "slot attention": 2,
    "transformer": 1,
    "autoregressive": 2,
    "causal": 1,
    "JEPA": 3,
}

def score_relevance(paper: dict) -> int:
    text = (paper["title"] + " " + paper["summary"]).lower()
    score = 0
    for kw, s in RELEVANCE_KEYWORDS.items():
        if kw in text:
            score += s
    return min(score, 20)


# ── Main ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Literature Pipeline for Agentic WM")
    parser.add_argument("--days", type=int, default=7, help="Days back to search")
    parser.add_argument("--max", type=int, default=30, help="Max results per query")
    parser.add_argument("--min-score", type=int, default=3, help="Min relevance score to save")
    parser.add_argument("--dry-run", action="store_true", help="Print only, don't save")
    args = parser.parse_args()
    
    all_papers = []
    
    print(f"🔍 Querying arXiv (last {args.days} days)...")
    for i, q in enumerate(QUERIES):
        print(f"  Query {i+1}/{len(QUERIES)}: {q[:80]}...")
        papers = query_arxiv(q, max_results=args.max, days_back=args.days)
        print(f"    → {len(papers)} results")
        all_papers.extend(papers)
        time.sleep(15)  # Rate limit (arXiv requires ~5s between requests, 15 to be safe)
    
    all_papers = deduplicate(all_papers)
    print(f"\n📄 {len(all_papers)} unique papers (after dedup)")
    
    # Score and filter
    for p in all_papers:
        p["relevance_score"] = score_relevance(p)
    
    all_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Print top results
    print(f"\n{'─'*80}")
    for p in all_papers[:15]:
        score = p["relevance_score"]
        marker = "🔥" if score >= 10 else ("⭐" if score >= 5 else "  ")
        print(f"  {marker} [{score:2d}] {p['title'][:80]}")
        print(f"        {p['url']}")
        print(f"        {p['published'][:10]} | {', '.join(p['authors'][:3])}")
        print()
    
    # Save high-relevance papers
    saved = 0
    for p in all_papers:
        if p["relevance_score"] >= args.min_score:
            date_str = p["published"][:10]
            safe_id = p["arxiv_id"].replace("/", "_")
            fname = f"{date_str}_{safe_id}.md"
            fpath = os.path.join(INBOX_DIR, fname)
            
            # Check if already exists
            if os.path.exists(fpath) and not args.dry_run:
                continue
            
            content = f"""# {p['title']}

- **arXiv:** [{p['arxiv_id']}]({p['url']})
- **Published:** {p['published'][:10]}
- **Authors:** {', '.join(p['authors'])}
- **Categories:** {', '.join(p['categories'])}
- **Relevance Score:** {p['relevance_score']}/20

## Abstract
{p['summary']}

## Quick Analysis
<!-- TODO: AI summary + comparison with Agentic WM framework -->
- [ ] Read abstract → rate actual relevance
- [ ] Check architecture: frame-as-token? decoupled action? tool use?
- [ ] Compare with Agentic WM framework
- [ ] Add to comparison matrix

"""
            if not args.dry_run:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w") as f:
                    f.write(content)
                saved += 1
    
    print(f"💾 Saved {saved} papers to {INBOX_DIR}/")
    
    if args.dry_run:
        print("   (dry-run mode, not saved)")
    
    # Print matrix placeholder
    print(f"\n{'─'*80}")
    print("Next: run 'AI summarize' step to fill in analysis for each saved paper.")
    print("Then update literature/matrix.md with comparison results.")


if __name__ == "__main__":
    main()
