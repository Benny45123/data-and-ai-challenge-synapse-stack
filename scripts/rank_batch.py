"""
scripts/rank_batch.py — Hackathon submission entry-point.

Constraints (from submission_spec.md):
  - CPU only, no GPU
  - No network access during ranking
  - ≤ 5 minutes wall-clock on the evaluation machine
  - ≤ 16 GB RAM
  - Output: CSV with columns [candidate_id, rank, score, reasoning]
  - Must produce exactly 100 ranked candidates

Architecture (all offline, all CPU):
  Stage 1 — Honeypot / trap detection      (hard filter, ~2s)
  Stage 2 — TF-IDF description scoring     (core signal, ~45s for 100K docs)
  Stage 3 — Feature engineering            (behavioral + career, ~10s)
  Stage 4 — Weighted composite ranking     (<1s)
  Stage 5 — Top-100 selection + CSV output (<1s)

KEY DESIGN DECISION:
  We score on career_history[].description text, NOT skills[] or current_title.
  Dataset analysis confirmed that skills/title are decorative noise:
    CAND_0004989: Title="Project Manager", Skills=[CNN,FAISS,...],
                  Description="Brand design and creative direction" → TRAP
    CAND_0000422: Title="AI Research Engineer", Skills=[MLflow,Photoshop,...],
                  Description="Built NLP pipelines... recommendation-style
                  features in production" → GENUINE

Usage:
  python scripts/rank_batch.py \\
      --candidates candidates.jsonl \\
      --output submission.csv

  python scripts/rank_batch.py \\
      --candidates sample_candidates.json \\
      --output submission.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from features.description_scorer import DescriptionScorer, build_description_corpus
from features.honeypot_detector import HoneypotDetector


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generator — grounded in description text, NOT skills/title
# ─────────────────────────────────────────────────────────────────────────────

# Production evidence snippets to surface in reasoning
_EVIDENCE_RE = re.compile(
    r'(?:'
    r'(?:deployed|shipped|launched|built|designed|implemented|led).{5,80}?'
    r'(?:production|users?|qps|scale|real|latency|ranking|retrieval|embedding|vector|search|ndcg|mrr)'
    r')',
    re.I,
)
_PRODUCTION_KEYWORDS = re.compile(
    r'\b(?:embed(?:ding)?s?|vector\s+(?:db|database|store|search)|retrieval|'
    r'semantic\s+search|ranking\s+(?:system|model)|recommendation\s+(?:system|engine)|'
    r'ndcg|mrr|hybrid\s+(?:search|retrieval)|fine[_\-\s]tun|a/b\s+test|'
    r'faiss|pinecone|qdrant|elasticsearch|learning\s+to\s+rank)\b',
    re.I,
)


def _extract_evidence_snippet(desc_text: str, max_len: int = 120) -> str:
    """Pull a specific evidence phrase from description text."""
    match = _EVIDENCE_RE.search(desc_text)
    if match:
        snippet = match.group(0).strip()
        return snippet[:max_len].strip()
    # Fall back to first sentence containing a relevant keyword
    for sentence in re.split(r'[.!?]', desc_text):
        if _PRODUCTION_KEYWORDS.search(sentence):
            return sentence.strip()[:max_len]
    return ""


def build_reasoning(
    candidate: dict,
    final_score: float,
    detail: dict,
    score_breakdown: dict,
) -> str:
    """
    Generate a 1-2 sentence reasoning grounded in actual description evidence.

    Format avoids title/skills (which are noise) and instead cites:
    - YoE and relevant role type from description
    - Specific production/technical evidence found in descriptions
    - Behavioral availability signal
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    yoe = profile.get("years_of_experience", 0)

    # Get most recent relevant role description
    desc_text = " ".join(e.get("description", "") for e in career)

    # Evidence snippet
    evidence = _extract_evidence_snippet(desc_text)

    # Behavioral summary
    rr = signals.get("recruiter_response_rate", 0.5)
    last_active = signals.get("last_active_date", "unknown")
    open_to_work = signals.get("open_to_work_flag", False)
    notice = signals.get("notice_period_days", 60)

    # Production hit count
    prod_hits = detail.get("production_hits", 0)

    # Build sentence 1: technical fit
    if evidence:
        tech_sentence = f"{yoe:.1f}yr exp; {evidence[:110]}."
    else:
        keyword_matches = _PRODUCTION_KEYWORDS.findall(desc_text)
        unique_kws = list(dict.fromkeys(kw.lower() for kw in keyword_matches))[:4]
        if unique_kws:
            tech_sentence = f"{yoe:.1f}yr exp; description evidence: {', '.join(unique_kws)}."
        else:
            tech_sentence = f"{yoe:.1f}yr exp; {prod_hits} production signals in career history."

    # Build sentence 2: availability
    avail_parts = []
    avail_parts.append(f"response_rate={rr:.0%}")
    avail_parts.append(f"last_active={last_active}")
    if open_to_work:
        avail_parts.append("open_to_work")
    if notice <= 30:
        avail_parts.append(f"notice={notice}d")

    avail_sentence = "; ".join(avail_parts) + "."

    return f"{tech_sentence} {avail_sentence}"


# ─────────────────────────────────────────────────────────────────────────────
# Candidate loader
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> List[dict]:
    """Load candidates from .jsonl, .jsonl.gz, or .json (array) format."""
    p = Path(path)
    print(f"[load] Reading {p.name} ({p.stat().st_size / 1e6:.1f} MB)…")
    t0 = time.time()
    candidates = []

    if p.suffix == ".gz" or path.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    elif p.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    elif p.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            candidates = data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    print(f"[load] Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Main ranking pipeline
# ─────────────────────────────────────────────────────────────────────────────

def rank(
    candidates: List[dict],
    top_k: int = 100,
    verbose: bool = True,
) -> List[Tuple[dict, float, dict, dict]]:
    """
    Full ranking pipeline.

    Returns:
        List of (candidate, final_score, detail, breakdown) sorted by score desc.
    """
    t_start = time.time()

    # ── Stage 1: Honeypot detection ───────────────────────────────────────────
    if verbose:
        print(f"\n[stage1] Honeypot detection on {len(candidates):,} candidates…")
    detector = HoneypotDetector()
    honeypot_flags: Dict[str, bool] = {}
    honeypot_count = 0
    clean_candidates = []
    for c in candidates:
        is_hp, reasons = detector.check(c)
        honeypot_flags[c["candidate_id"]] = is_hp
        if is_hp:
            honeypot_count += 1
        else:
            clean_candidates.append(c)

    if verbose:
        print(
            f"[stage1] Flagged {honeypot_count} honeypots → "
            f"{len(clean_candidates):,} clean candidates "
            f"({time.time()-t_start:.1f}s)"
        )

    # ── Stage 2: TF-IDF description scoring ───────────────────────────────────
    if verbose:
        print(f"\n[stage2] Building TF-IDF index over {len(clean_candidates):,} descriptions…")
    t2 = time.time()
    scorer = DescriptionScorer()
    corpus = build_description_corpus(clean_candidates)
    scorer.fit(corpus)
    if verbose:
        print(f"[stage2] TF-IDF fit complete ({time.time()-t2:.1f}s)")

    # ── Stage 3+4: Score all clean candidates ────────────────────────────────
    if verbose:
        print(f"\n[stage3] Scoring {len(clean_candidates):,} candidates…")
    t3 = time.time()
    scored: List[Tuple[dict, float, dict, dict]] = []
    for c in clean_candidates:
        final_score, detail = scorer.score(c)
        breakdown = _score_breakdown(c, detail)
        scored.append((c, final_score, detail, breakdown))

    if verbose:
        print(f"[stage3] Scoring complete ({time.time()-t3:.1f}s)")

    # ── Sort descending ────────────────────────────────────────────────────────
    scored.sort(key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"\n[rank] Total pipeline: {time.time()-t_start:.1f}s")
        _print_preview(scored[:10])

    return scored[:top_k]


def _score_breakdown(candidate: dict, detail: dict) -> dict:
    """Build a human-readable score breakdown for debugging."""
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    return {
        "yoe": profile.get("years_of_experience", 0),
        "title": profile.get("current_title", ""),
        "company": profile.get("current_company", ""),
        "tfidf": detail.get("tfidf", 0),
        "prod_hits": detail.get("production_hits", 0),
        "non_tech_hits": detail.get("non_tech_hits", 0),
        "behavioral": detail.get("behavioral", 0),
        "career_quality": detail.get("career_quality", 0),
        "disqualified": detail.get("disqualified", False),
        "disq_reason": detail.get("disq_reason", ""),
        "response_rate": signals.get("recruiter_response_rate", 0),
        "last_active": signals.get("last_active_date", ""),
        "open_to_work": signals.get("open_to_work_flag", False),
    }


def _print_preview(top10: List[Tuple]) -> None:
    print(f"\n{'Rank':<5} {'ID':<14} {'Score':<8} {'YoE':<6} "
          f"{'ProdHits':<10} {'TF-IDF':<8} {'Behav':<7} {'Title':<30}")
    print("-" * 95)
    for i, (c, score, detail, bd) in enumerate(top10, 1):
        print(
            f"#{i:<4} {c['candidate_id']:<14} {score:<8.4f} "
            f"{bd['yoe']:<6.1f} {bd['prod_hits']:<10} "
            f"{bd['tfidf']:<8.4f} {bd['behavioral']:<7.4f} "
            f"{bd['title'][:30]:<30}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CSV writer
# ─────────────────────────────────────────────────────────────────────────────

def write_submission_csv(
    ranked: List[Tuple[dict, float, dict, dict]],
    output_path: str,
) -> None:
    """
    Write the top-100 submission CSV.

    Columns (from sample_submission.csv):
        candidate_id, rank, score, reasoning
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (candidate, score, detail, breakdown) in enumerate(ranked, start=1):
            reasoning = build_reasoning(candidate, score, detail, breakdown)
            writer.writerow([
                candidate["candidate_id"],
                rank,
                f"{score:.4f}",
                reasoning,
            ])

    print(f"\n[output] Written {len(ranked)} rows → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Redrob hackathon: rank 100K candidates → top 100 CSV"
    )
    p.add_argument(
        "--candidates",
        default="candidates.jsonl",
        help="Path to candidates.jsonl / .jsonl.gz / sample_candidates.json",
    )
    p.add_argument(
        "--output",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of candidates to output (default: 100)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Write debug JSON alongside the CSV",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_wall = time.time()

    print("=" * 65)
    print("  Redrob Intelligent Candidate Ranker — Hackathon Submission")
    print("=" * 65)

    # Load
    if not os.path.exists(args.candidates):
        print(f"ERROR: {args.candidates} not found.")
        sys.exit(1)

    candidates = load_candidates(args.candidates)
    total = len(candidates)

    # Rank
    ranked = rank(candidates, top_k=args.top_k, verbose=True)

    # Write CSV
    write_submission_csv(ranked, args.output)

    # Optional debug output
    if args.debug:
        debug_path = args.output.replace(".csv", "_debug.json")
        debug_data = [
            {
                "rank": i + 1,
                "candidate_id": c["candidate_id"],
                "score": score,
                "breakdown": bd,
                "detail": detail,
            }
            for i, (c, score, detail, bd) in enumerate(ranked)
        ]
        with open(debug_path, "w") as f:
            json.dump(debug_data, f, indent=2)
        print(f"[debug] Written debug info → {debug_path}")

    wall_time = time.time() - t_wall
    print(f"\n{'='*65}")
    print(f"  Done in {wall_time:.1f}s | Pool: {total:,} | Output: top {len(ranked)}")
    print(f"  → {args.output}")
    print(f"{'='*65}\n")

    if wall_time > 270:
        print(f"⚠ WARNING: {wall_time:.0f}s elapsed — approaching 5-min limit on eval machine.")


if __name__ == "__main__":
    main()
