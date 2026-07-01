"""
features/description_scorer.py — Score candidates from career description text.

KEY INSIGHT FROM DATASET ANALYSIS:
  skills[] and current_title are DECORATIVE NOISE — they are randomly assigned
  in this dataset and do not reflect actual work done.

  career_history[].description is the GROUND TRUTH signal.

  Evidence:
    CAND_0004989 — Title: "Project Manager", Skills: [Kubernetes, CNN, FAISS, ...]
                   Description: "Brand design and creative direction... packaging design"
                   → TRAP candidate; should rank near 0.

    CAND_0000422 — Title: "AI Research Engineer", Skills: [MLflow, Photoshop, ...]
                   Description: "Built NLP pipelines... recommendation-style features
                                 at a product company... production"
                   → GENUINE candidate; should rank high.

Scoring pipeline:
  1. TF-IDF cosine similarity between JD query and concatenated descriptions
  2. Production evidence keyword bonus (deployed, served Xk QPS, etc.)
  3. Retrieval/ranking domain signals (NDCG, hybrid search, embedding service)
  4. Non-technical content penalty (brand design, sales, fulfillment, etc.)
  5. Disqualifier detection (consulting-only, pure research, CV/speech primary)
"""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── JD Query ─────────────────────────────────────────────────────────────────
# Constructed from the full JD text (job_description.docx).
# Includes both explicit requirements and implied role signals.

JD_QUERY = """
production embeddings based retrieval deployed sentence transformers OpenAI embeddings
BGE E5 embedding drift index refresh retrieval quality regression
vector database hybrid search infrastructure pinecone weaviate qdrant milvus
opensearch elasticsearch faiss operational experience
python code quality production systems
evaluation framework ranking NDCG MRR MAP offline online correlation A/B test interpretation
recommendation engine ranking system search system shipped real users scale
learning to rank LTR XGBoost neural ranker
LLM fine-tuning LoRA QLoRA PEFT
startup product company founding team
reranking cross encoder bi encoder dense retrieval sparse retrieval BM25
embedding service semantic search information retrieval
MLOps model serving latency throughput QPS
mentor engineering team architecture decision
""".strip()

# ── Production signal patterns ────────────────────────────────────────────────
# These patterns in description text indicate real shipped systems.

_PRODUCTION_PATTERNS = [
    # Deployed at scale
    r'\b(?:deployed|shipped|launched|released|rolled\s+out)\b',
    r'\b(?:served|serving|handles?|processing)\b.{0,40}\b(?:\d+[kmb]?\s*(?:qps|rps|req|requests?|users?)|real\s+users?|production)\b',
    r'\bproduction\b.{0,60}\b(?:system|service|pipeline|endpoint|api|model)\b',
    # Retrieval specific
    r'\b(?:hybrid|dense|sparse)\s+(?:search|retrieval)\b',
    r'\bembedding\s+(?:service|pipeline|index|store|drift)\b',
    r'\bvector\s+(?:search|store|database|index|db)\b',
    r'\b(?:faiss|pinecone|qdrant|weaviate|milvus|opensearch|elasticsearch)\b',
    r'\b(?:sentence[_\-\s]transformers?|bge|e5\b|openai\s+embed)\b',
    # Ranking / eval
    r'\b(?:ndcg|mrr|map@\d|precision@\d|recall@\d)\b',
    r'\b(?:learning[_\-\s]to[_\-\s]rank|ltr|lambdamart|ranknet)\b',
    r'\b(?:a/b\s+test(?:ing)?|experiment(?:ing)?|online\s+eval)\b',
    r'\b(?:recommendation\s+(?:system|engine)|ranker|ranking\s+model)\b',
    # Operational
    r'\b(?:latency|throughput|p99|p95|sla|uptime)\b',
    r'\b(?:fine[_\-\s]tun(?:ed?|ing)|lora|qlora|peft)\b',
    r'\b(?:mlops|model\s+(?:serving|registry|versioning|deploy))\b',
]
_PROD_RE = [re.compile(p, re.I) for p in _PRODUCTION_PATTERNS]

# ── Non-technical content patterns (trap signals) ────────────────────────────
# If descriptions are predominantly about these topics, the candidate's
# AI skill tags are keyword stuffing.

_NON_TECH_PATTERNS = [
    r'\b(?:brand\s+(?:design|identity)|visual\s+(?:system|identity)|typography|packaging\s+design)\b',
    r'\b(?:creative\s+direction|adobe\s+suite|photoshop|illustrator|indesign)\b',
    r'\b(?:operations\s+(?:management|lead)|warehouse|fulfillment|logistics\s+(?:company|platform))\b',
    r'\b(?:sales\s+(?:cycle|quota|pipeline|prospecting)|carry(?:ing)?\s+a?\s*\$[\d\.]+[mk]?\s*arr)\b',
    r'\b(?:demand\s+generation|seo|paid\s+acquisition|content\s+marketing|email\s+nurture)\b',
    r'\b(?:customer\s+support|support\s+(?:tickets?|agents?)|knowledge\s+base|tier[_\-\s]1)\b',
    r'\b(?:mechanical\s+engineering|dfm|dfma|manufacturing|cad|prototype|tooling)\b',
    r'\b(?:android|ios\s+(?:app|development)|swift|kotlin|push\s+notification|shopping\s+flow)\b',
    r'\b(?:financial\s+(?:reporting|statements?|modelling)|balance\s+sheet|audit|accounting)\b',
    r'\b(?:hr\s+(?:operations|processes?)|payroll|talent\s+acquisition\s+ops)\b',
    r'\b(?:civil\s+engineering|construction|infrastructure\s+projects?)\b',
    r'\b(?:six\s+sigma|lean\s+manufacturing|process\s+improvement|supply\s+chain)\b',
]
_NON_TECH_RE = [re.compile(p, re.I) for p in _NON_TECH_PATTERNS]

# ── Consulting-only firms (JD-specified disqualifier) ────────────────────────
_CONSULTING_FIRMS = frozenset({
    'tcs', 'tata consultancy', 'infosys', 'wipro', 'accenture',
    'cognizant', 'capgemini', 'hcl', 'hcl technologies', 'tech mahindra',
    'mphasis', 'hexaware', 'mindtree', 'l&t infotech', 'ltimindtree',
    'birlasoft', 'mastech', 'niit technologies', 'persistent systems',
})

# ── Pure research signals ────────────────────────────────────────────────────
_RESEARCH_PATTERNS = [
    r'\b(?:authored?\s+paper|published|arxiv|neurips|icml|iclr|acl|emnlp)\b',
    r'\b(?:ablation\s+study|research\s+paper|literature\s+review|phd\s+thesis)\b',
    r'\b(?:academic\s+lab|university\s+research|postdoc|research\s+scientist\s+at\s+university)\b',
]
_RESEARCH_RE = [re.compile(p, re.I) for p in _RESEARCH_PATTERNS]

# ── CV / Speech / Robotics primary (without NLP/IR) ─────────────────────────
_CV_SPEECH_RE = [
    re.compile(r'\b(?:object\s+detection|yolo|resnet|image\s+(?:classification|segmentation)|cnn\s+for\s+image)\b', re.I),
    re.compile(r'\b(?:speech\s+recognition|asr|tts|text[_\-\s]to[_\-\s]speech|speech\s+synthesis)\b', re.I),
    re.compile(r'\b(?:robotics|ros\b|slam\b|motion\s+planning|autonomous\s+(?:vehicle|driving))\b', re.I),
]
_NLP_IR_RE = re.compile(
    r'\b(?:nlp|natural\s+language|text\s+(?:classification|retrieval)|information\s+retrieval|'
    r'embeddings?|transformer|bert|sentence[_\-\s]bert|search|ranking|recommendation)\b', re.I
)

# ── Last active date baseline ────────────────────────────────────────────────
_TODAY = date(2026, 6, 30)


# ─────────────────────────────────────────────────────────────────────────────

class DescriptionScorer:
    """
    TF-IDF based scorer over career description text.

    Usage:
        scorer = DescriptionScorer()
        scorer.fit(id_to_desc_map)   # build TF-IDF corpus
        score = scorer.score(candidate_dict)
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=40_000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            analyzer="word",
        )
        self._corpus_matrix = None
        self._corpus_ids: List[str] = []
        self._tfidf_scores: Dict[str, float] = {}

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, id_to_desc: Dict[str, str]) -> None:
        """
        Build the TF-IDF index over all description texts.

        Args:
            id_to_desc: {candidate_id: concatenated_career_description}
        """
        self._corpus_ids = list(id_to_desc.keys())
        corpus = list(id_to_desc.values())

        self._corpus_matrix = self._vectorizer.fit_transform(corpus)

        # Project JD query into same space
        jd_vec = self._vectorizer.transform([JD_QUERY])

        # Compute cosine similarity for all docs in one matrix op
        sims = cosine_similarity(jd_vec, self._corpus_matrix).flatten()

        self._tfidf_scores = {
            cid: float(sim) for cid, sim in zip(self._corpus_ids, sims)
        }

    # ── Score one candidate ───────────────────────────────────────────────────

    def score(self, candidate: dict) -> Tuple[float, dict]:
        """
        Compute a composite score for one candidate.

        Returns:
            (final_score_0_to_1, detail_dict_for_reasoning)
        """
        cid = candidate["candidate_id"]
        career = candidate.get("career_history", [])
        desc_text = " ".join(e.get("description", "") for e in career)
        signals = candidate.get("redrob_signals", {})

        # ── A: TF-IDF semantic match ──────────────────────────────────────────
        tfidf_score = self._tfidf_scores.get(cid, 0.0)

        # ── B: Production evidence bonus ──────────────────────────────────────
        prod_hits = sum(1 for r in _PROD_RE if r.search(desc_text))
        production_score = min(1.0, prod_hits / 8.0)

        # ── C: Non-technical content penalty ─────────────────────────────────
        non_tech_hits = sum(1 for r in _NON_TECH_RE if r.search(desc_text))
        non_tech_penalty = min(1.0, non_tech_hits * 0.35)

        # ── D: Behavioral multiplier (availability + reliability) ─────────────
        behavioral_score = self._behavioral_score(candidate, signals)

        # ── E: Disqualifier detection ─────────────────────────────────────────
        disqualifier, disq_reason = self._check_disqualifiers(candidate, desc_text)

        # ── F: Career quality (YoE band, company type) ───────────────────────
        career_quality = self._career_quality(candidate)

        # ── Composite ─────────────────────────────────────────────────────────
        base = (
            tfidf_score      * 0.40 +
            production_score * 0.30 +
            career_quality   * 0.15 +
            behavioral_score * 0.15
        ) - non_tech_penalty * 0.40

        # Disqualifiers hard-cap the score
        final = max(0.0, base) * (0.0 if disqualifier else 1.0)

        detail = {
            "tfidf": round(tfidf_score, 4),
            "production_hits": prod_hits,
            "non_tech_hits": non_tech_hits,
            "behavioral": round(behavioral_score, 4),
            "career_quality": round(career_quality, 4),
            "disqualified": disqualifier,
            "disq_reason": disq_reason,
        }
        return round(min(1.0, final), 6), detail

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _behavioral_score(self, candidate: dict, signals: dict) -> float:
        """
        Availability × reliability composite from redrob_signals.

        A perfect-on-paper candidate who is inactive / unresponsive is
        effectively unavailable. (JD note, line 95-96)
        """
        rr = signals.get("recruiter_response_rate", 0.5)
        icr = signals.get("interview_completion_rate", 0.8)
        open_to_work = signals.get("open_to_work_flag", False)

        # Recency: days since last active
        try:
            last_active = date.fromisoformat(signals.get("last_active_date", "2025-01-01"))
            days_inactive = (_TODAY - last_active).days
        except ValueError:
            days_inactive = 180

        # Sigmoid decay: 0d inactive=1.0, 30d=0.85, 90d=0.60, 180d=0.35
        recency = 1.0 / (1.0 + (days_inactive / 60.0) ** 1.5)

        # Notice period bonus: <30 days is preferred (JD line 75)
        notice = signals.get("notice_period_days", 60)
        notice_bonus = 0.1 if notice <= 30 else (0.05 if notice <= 60 else 0.0)

        # GitHub activity score (-1 = no github, 0-100)
        gh = signals.get("github_activity_score", -1)
        github_bonus = (gh / 100.0) * 0.1 if gh >= 0 else 0.0

        score = (
            rr * 0.35 +
            recency * 0.30 +
            icr * 0.20 +
            (0.15 if open_to_work else 0.05) +
            notice_bonus +
            github_bonus
        )
        return min(1.0, score)

    def _career_quality(self, candidate: dict) -> float:
        """YoE band fit, company-type quality, shipper vs researcher signal."""
        yoe = candidate["profile"].get("years_of_experience", 0)
        career = candidate.get("career_history", [])

        # YoE band: 5-9 is ideal; outside band decays gracefully (JD line 30)
        if 5 <= yoe <= 9:
            yoe_score = 1.0
        elif 4 <= yoe < 5 or 9 < yoe <= 11:
            yoe_score = 0.75
        elif 3 <= yoe < 4 or 11 < yoe <= 13:
            yoe_score = 0.45
        else:
            yoe_score = 0.20

        # Company type: product company > mixed > consulting-only
        companies = [e.get("company", "").lower() for e in career]
        consulting_count = sum(
            1 for c in companies
            if any(firm in c for firm in _CONSULTING_FIRMS)
        )
        product_count = len(companies) - consulting_count
        if product_count > 0:
            company_score = min(1.0, product_count / len(companies) + 0.3)
        else:
            company_score = 0.2   # consulting-only is a soft disqualifier

        # Company size: startup experience valued (JD line 83-85)
        sizes = [e.get("company_size", "") for e in career]
        startup_exp = any(s in ("1-10", "11-50", "51-200", "201-500") for s in sizes)
        startup_bonus = 0.1 if startup_exp else 0.0

        return min(1.0, yoe_score * 0.5 + company_score * 0.4 + startup_bonus)

    def _check_disqualifiers(
        self, candidate: dict, desc_text: str
    ) -> Tuple[bool, str]:
        """
        Hard disqualifiers from JD (lines 33-70).
        Returns (is_disqualified, reason).
        """
        career = candidate.get("career_history", [])
        companies = [e.get("company", "").lower() for e in career]

        # 1. Consulting-only entire career (JD line 65-66)
        if companies and all(
            any(firm in c for firm in _CONSULTING_FIRMS) for c in companies
        ):
            return True, "consulting_only_career"

        # 2. Pure CV/Speech/Robotics without NLP/IR (JD line 67-68)
        cv_speech_hits = sum(1 for r in _CV_SPEECH_RE if r.search(desc_text))
        if cv_speech_hits >= 2 and not _NLP_IR_RE.search(desc_text):
            return True, "cv_speech_robotics_without_nlp"

        # 3. Zero production evidence + strong research signals
        prod_hits = sum(1 for r in _PROD_RE if r.search(desc_text))
        research_hits = sum(1 for r in _RESEARCH_RE if r.search(desc_text))
        if research_hits >= 2 and prod_hits == 0:
            return True, "pure_research_no_production"

        # 4. Predominantly non-technical work (trap candidate)
        non_tech_hits = sum(1 for r in _NON_TECH_RE if r.search(desc_text))
        if non_tech_hits >= 3:
            return True, "non_technical_work_primary"

        return False, ""


def build_description_corpus(candidates: List[dict]) -> Dict[str, str]:
    """
    Build {candidate_id: description_text} for TF-IDF fitting.
    Concatenates all career history descriptions per candidate.
    """
    corpus = {}
    for c in candidates:
        cid = c["candidate_id"]
        desc = " ".join(
            e.get("description", "") for e in c.get("career_history", [])
        )
        # Append summary for additional context (but weight descriptions more)
        summary = c.get("profile", {}).get("summary", "")
        corpus[cid] = f"{desc} {summary}".strip()
    return corpus
