"""
features/honeypot_detector.py — Detect honeypot and trap candidates.

From README: "~80 honeypots with subtly impossible profiles. Submissions with
honeypot rate > 10% in top 100 are DISQUALIFIED."

From dataset scan (100K candidates):
  - 48 candidates with YoE vs career_history duration mismatch (hard flag)
  - 21 candidates with 'expert' skill at 0 duration (soft flag)
  - 2,888 impossible age candidates (soft flag; threshold needs calibration)

Trap types (from JD hackathon note, line 91-96):
  - Keyword stuffers: AI skill tags but non-technical descriptions
  - Plain-language Tier 5s: genuine candidates without AI buzzwords (NOT traps!)
  - Behavioral twins: similar behavioral signals, very different actual fit
  - Honeypots: internally impossible / contradictory profile data
"""
from __future__ import annotations

import re
from datetime import date
from typing import List, Tuple

_TODAY = date(2026, 6, 30)

# Consulting-firm list shared with description_scorer
_CONSULTING_FIRMS = frozenset({
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'hcl technologies', 'tech mahindra',
    'mphasis', 'hexaware', 'mindtree', 'l&t infotech', 'ltimindtree',
    'birlasoft', 'mastech', 'niit technologies', 'persistent systems',
})

# These AI skill labels, when appearing WITHOUT description evidence,
# are the hallmark of keyword-stuffing trap candidates.
_AI_SKILL_TAGS = frozenset({
    'embeddings', 'vector search', 'semantic search', 'rag', 'llms',
    'faiss', 'pinecone', 'qdrant', 'weaviate', 'fine-tuning llms',
    'langchain', 'prompt engineering', 'hugging face transformers',
    'sentence transformers', 'information retrieval',
})

# Description evidence that marks a candidate as a real AI/ML practitioner
_REAL_AI_DESC_RE = re.compile(
    r'\b(?:embed(?:ding)?|retrieval|vector\s+(?:db|database|store|search)|'
    r'recommendation|ranking|nlp|natural\s+language|transformer|bert|'
    r'fine[_\-\s]tun|learning\s+to\s+rank|semantic\s+search|'
    r'ndcg|mrr|a/b\s+test|rerank|hybrid\s+search|dense\s+retrieval)\b',
    re.I,
)


class HoneypotDetector:
    """
    Returns is_honeypot flag and a list of reasons for each candidate.

    Call check() on each candidate dict.
    Returns (is_honeypot: bool, reasons: list[str])
    """

    def check(self, candidate: dict) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        reasons += self._check_yoe_mismatch(candidate)
        reasons += self._check_expert_zero_duration(candidate)
        reasons += self._check_impossible_age(candidate)
        reasons += self._check_skill_description_mismatch(candidate)
        reasons += self._check_assessment_phantoms(candidate)

        # Hard honeypot: any hard flag → disqualify
        hard_flags = [r for r in reasons if r.startswith("HARD:")]
        soft_flags = [r for r in reasons if r.startswith("SOFT:")]

        is_honeypot = bool(hard_flags) or len(soft_flags) >= 2

        return is_honeypot, reasons

    # ── Check 1: YoE vs career history duration ───────────────────────────────

    def _check_yoe_mismatch(self, candidate: dict) -> List[str]:
        yoe = candidate["profile"].get("years_of_experience", 0)
        total_months = sum(
            e.get("duration_months", 0) for e in candidate.get("career_history", [])
        )
        if yoe <= 0:
            return []
        ratio = total_months / (yoe * 12)
        # Real inconsistency threshold: ratio > 1.6 or < 0.4 (beyond normal rounding)
        if ratio > 1.6 or ratio < 0.4:
            return [f"HARD:yoe_mismatch(claimed={yoe},career_months={total_months},ratio={ratio:.2f})"]
        return []

    # ── Check 2: Expert proficiency with near-zero usage duration ─────────────

    def _check_expert_zero_duration(self, candidate: dict) -> List[str]:
        flags = []
        for skill in candidate.get("skills", []):
            if (
                skill.get("proficiency") == "expert"
                and skill.get("duration_months") is not None
                and skill["duration_months"] < 2
            ):
                flags.append(f"SOFT:expert_zero_duration({skill['name']})")
        # Only raise if 2+ skills show this pattern (one could be a data error)
        return flags if len(flags) >= 2 else []

    # ── Check 3: Impossible age (started work before 16) ─────────────────────

    def _check_impossible_age(self, candidate: dict) -> List[str]:
        yoe = candidate["profile"].get("years_of_experience", 0)
        edu = candidate.get("education", [])
        if not edu:
            return []
        earliest_start = min(e.get("start_year", 2000) for e in edu)
        implied_birth = earliest_start - 18
        implied_age = _TODAY.year - implied_birth
        # Working since before age 12 is impossible
        if implied_age - yoe < 12:
            return [
                f"SOFT:impossible_age(edu_start={earliest_start},"
                f"implied_age={implied_age},yoe={yoe})"
            ]
        return []

    # ── Check 4: AI skill tags with zero description evidence ─────────────────

    def _check_skill_description_mismatch(self, candidate: dict) -> List[str]:
        skill_names = {s["name"].lower() for s in candidate.get("skills", [])}
        has_ai_tags = bool(skill_names & _AI_SKILL_TAGS)

        if not has_ai_tags:
            return []

        desc_text = " ".join(
            e.get("description", "") for e in candidate.get("career_history", [])
        )
        has_desc_evidence = bool(_REAL_AI_DESC_RE.search(desc_text))

        if not has_desc_evidence:
            return ["SOFT:ai_skill_tags_without_description_evidence"]
        return []

    # ── Check 5: Phantom skill assessments ────────────────────────────────────

    def _check_assessment_phantoms(self, candidate: dict) -> List[str]:
        """
        A high assessment score for a skill that appears nowhere in the profile
        (description, summary, or skill tags) is a phantom signal.
        """
        full_text = (
            candidate.get("profile", {}).get("summary", "") + " " +
            " ".join(e.get("description", "") for e in candidate.get("career_history", [])) + " " +
            " ".join(s["name"] for s in candidate.get("skills", []))
        ).lower()

        phantom_count = 0
        for skill_name, score in candidate.get("redrob_signals", {}).get(
            "skill_assessment_scores", {}
        ).items():
            if score >= 85 and skill_name.lower() not in full_text:
                phantom_count += 1

        if phantom_count >= 2:
            return [f"SOFT:phantom_assessment_scores(count={phantom_count})"]
        return []
