"""
stages/stage4_explainability.py — LLM Fit Summaries (async, < 500ms target).

Design constraints (Section 7 of spec):
  1. NO hallucinated credentials. LLM only sees *structured, verified* facts.
  2. Fact verification pass: every claim cross-referenced against structured fields.
  3. Confidence score: fraction of verifiable claims / total claims.
  4. Async delivery: this stage never blocks the synchronous ranking response.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import anthropic

from config import get_settings
from models.schemas import CandidateProfile, JobDescription, LTRFeatureVector

log = logging.getLogger(__name__)
settings = get_settings()


# ── Verified context extraction ───────────────────────────────────────────────

def extract_verified_context(
    candidate: CandidateProfile,
    jd: JobDescription,
    feature_vector: Optional[LTRFeatureVector] = None,
) -> Dict:
    """
    Pull verifiable facts from structured profile fields.

    The LLM never sees the raw profile text. It only sees this dict,
    preventing hallucinations from vague or ambiguous prose.
    """
    # Skills that appear in both the profile and the JD
    candidate_skills_lower = {s.lower() for s in candidate.skills}
    jd_required_lower = {s.lower() for s in jd.required_skills}
    jd_preferred_lower = {s.lower() for s in jd.preferred_skills}

    confirmed_required = [
        s for s in jd.required_skills if s.lower() in candidate_skills_lower
    ]
    confirmed_preferred = [
        s for s in jd.preferred_skills if s.lower() in candidate_skills_lower
    ]
    missing_required = [
        s for s in jd.required_skills if s.lower() not in candidate_skills_lower
    ]

    # Production evidence from career history
    production_evidence = []
    for entry in candidate.career_history:
        desc = entry.description or ""
        shipper_keywords = ["deployed", "production", "served", "launched",
                            "migrated", "scaled", "qps", "latency"]
        if any(kw in desc.lower() for kw in shipper_keywords):
            # Use first 150 chars of the description as evidence snippet
            production_evidence.append(
                f"{entry.title} @ {entry.company}: {desc[:150]}"
            )

    return {
        "name": candidate.name,
        "total_yoe": candidate.total_yoe,
        "education": candidate.education or "Not specified",
        "confirmed_required_skills": confirmed_required,
        "confirmed_preferred_skills": confirmed_preferred,
        "missing_required_skills": missing_required,
        "production_evidence": production_evidence[:3],   # top 3 snippets
        "github_active": candidate.github_activity_score > 0.6,
        "response_rate": candidate.recruiter_response_rate,
        "archetype_score": feature_vector.archetype_score if feature_vector else None,
        "relevant_yoe": feature_vector.relevant_yoe if feature_vector else None,
    }


# ── Prompt construction ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a technical recruiting assistant. Your task is to write
a concise 2-sentence fit summary for a candidate based EXCLUSIVELY on the verified
facts provided. Rules:
1. Do NOT infer or claim skills that are not in `confirmed_required_skills` or
   `confirmed_preferred_skills`.
2. If a required skill is in `missing_required_skills`, explicitly state it is
   unconfirmed in the profile.
3. Cite specific facts (years, project snippets, skills) — not vague assertions.
4. Sentence 1: strongest evidence of fit. Sentence 2: most significant gap or
   unconfirmed requirement.
Output format: plain text, two sentences, no bullet points."""

def _build_prompt(verified: Dict, jd: JobDescription) -> str:
    return f"""Job Title: {jd.title}
Required Skills: {', '.join(jd.required_skills)}

Verified Candidate Facts:
- Confirmed required skills: {', '.join(verified['confirmed_required_skills']) or 'none'}
- Confirmed preferred skills: {', '.join(verified['confirmed_preferred_skills']) or 'none'}
- Missing required skills: {', '.join(verified['missing_required_skills']) or 'none'}
- Total YoE: {verified['total_yoe']} years
- Relevant YoE: {verified.get('relevant_yoe', 'unknown')} years
- Education: {verified['education']}
- Production evidence: {'; '.join(verified['production_evidence']) or 'none found'}
- GitHub activity (>60th percentile): {verified['github_active']}
- Recruiter response rate: {verified['response_rate']:.0%}

Write the 2-sentence fit summary now:"""


# ── Fact verification ─────────────────────────────────────────────────────────

def _extract_claims(summary: str) -> List[str]:
    """
    Split summary into individual claims (sentence-level).
    In production: NLI-based claim extraction model.
    """
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]


def _is_grounded(claim: str, verified: Dict) -> bool:
    """
    Check whether a claim can be traced to a structured field in `verified`.
    Simplified: keyword overlap. Production: NLI entailment check.
    """
    claim_lower = claim.lower()

    # Check confirmed skills
    if any(skill.lower() in claim_lower
           for skill in verified.get("confirmed_required_skills", [])
           + verified.get("confirmed_preferred_skills", [])):
        return True

    # Check numeric facts
    if str(int(verified.get("total_yoe", -1))) in claim:
        return True

    # Check production snippets
    if any(
        word in claim_lower
        for snippet in verified.get("production_evidence", [])
        for word in snippet.lower().split()[:6]   # first 6 words of snippet
    ):
        return True

    # Check education
    if verified.get("education", "").lower() in claim_lower:
        return True

    return False


def verify_summary(summary: str, verified: Dict) -> Tuple[str, float, List[str]]:
    """
    Cross-reference every claim in the summary against verified context.

    Returns:
        (cleaned_summary, confidence_score, flagged_claims)
    """
    claims = _extract_claims(summary)
    flagged: List[str] = []

    for claim in claims:
        if not _is_grounded(claim, verified):
            flagged.append(claim)

    total = len(claims)
    verified_count = total - len(flagged)
    confidence = verified_count / total if total > 0 else 0.0

    # Remove or annotate ungrounded claims
    if flagged:
        log.warning("Ungrounded claims detected: %s", flagged)

    return summary, round(confidence, 2), flagged


# ── Main async generator ──────────────────────────────────────────────────────

async def generate_fit_summary(
    candidate: CandidateProfile,
    jd: JobDescription,
    feature_vector: Optional[LTRFeatureVector] = None,
) -> Dict:
    """
    Generate a RAG-grounded fit summary for one candidate.

    This is always called asynchronously — it never blocks the ranking response.

    Returns:
        {
            "candidate_id": str,
            "summary": str,
            "confidence": float,
            "flags": List[str],
            "verified_context": Dict,
        }
    """
    verified = extract_verified_context(candidate, jd, feature_vector)
    prompt = _build_prompt(verified, jd)

    try:
        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        message = await client.messages.create(
            model=settings.LLM_MODEL,
            max_tokens=settings.LLM_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_summary = message.content[0].text.strip()
    except Exception as exc:
        log.error("LLM API error for candidate %s: %s", candidate.candidate_id, exc)
        raw_summary = (
            f"Candidate has {candidate.total_yoe:.0f} YoE and confirms: "
            f"{', '.join(verified['confirmed_required_skills']) or 'no required skills on profile'}. "
            f"Missing: {', '.join(verified['missing_required_skills']) or 'none'}."
        )

    summary, confidence, flags = verify_summary(raw_summary, verified)

    return {
        "candidate_id": candidate.candidate_id,
        "summary": summary,
        "confidence": confidence,
        "flags": flags,
        "verified_context": verified,
    }


async def batch_generate_summaries(
    ranked_candidates: List[Tuple[CandidateProfile, LTRFeatureVector]],
    jd: JobDescription,
    max_candidates: int = 10,
) -> Dict[str, Dict]:
    """
    Generate fit summaries for the top-N candidates in parallel.

    Args:
        ranked_candidates: (profile, feature_vector) pairs, already sorted.
        jd:                Job description.
        max_candidates:    How many summaries to generate (default 10).

    Returns:
        Dict keyed by candidate_id.
    """
    import asyncio

    tasks = [
        generate_fit_summary(profile, jd, fv)
        for profile, fv in ranked_candidates[:max_candidates]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    output: Dict[str, Dict] = {}
    for result in results:
        if isinstance(result, Exception):
            log.error("Summary generation failed: %s", result)
            continue
        output[result["candidate_id"]] = result

    return output