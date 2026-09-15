"""
Stage 3: LLM Scoring

Scores every Standard Concept candidate from Stage 2 against the source term
with an LLM, following OMOP hierarchy rules (Equivalent > Parent; Child and
meaning-changed concepts are rejected), and ranks candidates by that score.

The LLM backend is provider-agnostic via LLMClient (OpenAI, Together AI).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..utils import sigmoid_normalize
from ..llm_client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt templates
# =============================================================================

SYSTEM_PROMPT = """You are a clinical terminology and ontology expert. You are given one entity and several candidate OMOP CDM concepts.
Your task is to evaluate semantic equivalence using strict hierarchy logic.
Never select a broader (parent) concept when a semantically equivalent candidate exists.
Score EACH candidate on a scale of 0.0-5.0 based strictly on clinical meaning, not wording similarity."""

USER_PROMPT_TEMPLATE = """
### Mapping Rules
1. Semantic Equivalence First
- If a candidate has the SAME clinical meaning as the entity (even if wording differs), you MUST select it and score 5.

2. Parent (Broader) Concept — Allowed IF AND ONLY IF there is NO semantic equivalence candidate
- If no semantically equivalent candidate exists, you MAY select a MORE GENERAL (higher-level, fully contains the entity meaning) concept.

3. Child (Narrower) Concept — STRICTLY FORBIDDEN
- A "child" concept is more specific than the entity (e.g., adding a body site, anatomical structure, severity, or underlying cause NOT present in the entity).
- The entity MUST NEVER be mapped to a child concept.
- Such candidates introduce new meaning and are always invalid.

### IMPORTANT RULES
- Do NOT guess: Choosing a wrong concept with additional or different meaning is worse than choosing none.
- Semantic over Lexical: Scoring is based on clinical meaning, NOT string similarity.
- Original_concept_name: Some candidates include an original non-standard concept name (mapped to the standard concept via "Maps to").
  a) If this original name is semantically equivalent to the entity,
     that candidate MUST receive a score ≥ 4.5 (unless the standard concept itself introduces
     a clear meaning change that cannot be justified). Do NOT ignore this field.
- Drug / Measurement / Procedure Concepts:
  a) FIRST, normalize equivalent expressions before comparing(MANDATORY MATHEMATICAL CONVERSION):
    percentage concentrations and mg/ml are interchangeable using explicit mathematical conversion:
    1% = 1 g / 100 ml = 10 mg/ml
    therefore X% = X x 10 mg/ml (e.g., 2% = 20 mg/ml),
    total dose / volume = concentration (e.g., 4 mg/2 ml = 2 mg/ml).
    Additional normalization rule for dose/volume notation:
    Expressions written as "X mg / Y mL" encode BOTH total dose and concentration.
    Example:
    5 mg / 1 mL → total dose = 5 mg, volume = 1 mL, concentration = 5 mg/mL
    After normalization, if quantitative meaning is identical, they MUST be treated as Equivalent. Do NOT compare raw numeric strings without unit conversion.
  b) THEN, compare ALL specifications
    eg. active ingredient + normalized strength + form + volume.
    eg. anotomic site + view
  c) A candidate matching ALL specifications to entity MUST score higher than one that omits volume (omitting = broader/parent).
   If the entity specifies total volume (explicitly or via X mg/Y mL notation), a candidate omitting volume is a Parent concept, not Equivalent.
  d) Manufacturer/brand names and packaging info (e.g., "by Hospira", "box of 5")
    are ADDITIONAL details. A candidate with such extras is a child candidate and MUST score LOWER than
    the equivalent unbranded/unpackaged candidate.
    Prefer the candidate WITHOUT brand or packaging information when core specifications are identical (branded versions are child concepts due to added attributes).
    Device specifications (e.g., prefilled syringe, vial, cartridge, pen) are additional attributes and therefore represent Child concepts.
    e) Compare quantitative MEANING after normalization — not raw numbers.
     Same number IS NOT same meaning (e.g., 2% IS NOT 2 mL).
     Convert to comparable units (e.g., %, mg/mL) and check if the clinical quantity is identical.

- Procedure concepts must also be compared using defining attributes such as:
  procedure method, body site, laterality, imaging modality, and imaging view.

- Extra clinical specification = Child concept.
  Missing clinical specification = Parent concept.

- For imaging procedures, view/projection is a defining attribute.
  If a candidate introduces an additional imaging view not present in the entity
  (e.g., PA + lateral vs lateral),
  the candidate is a Child (Narrower) concept.


### Decision Process (STRICT HIERARCHY LOCK — ENHANCED)

0. Mandatory Equivalence Check (Pre-Classification)
   BEFORE classification and scoring, the model MUST determine
   whether a fully Equivalent candidate exists.

   - Equivalent requires matching ALL defining attributes after normalization
     (ingredient, strength, concentration, total volume, dose, form, route,
      anatomic site, severity, histologic subtype, and intent when applicable).
   - Omission of a defining attribute disqualifies equivalence.
   - Alteration of a defining attribute disqualifies equivalence.

   Hard constraints:
   - 2 mL vs 1 mL = Meaning-changed (NOT Parent, NOT Child).
   - Any explicit numeric mismatch (volume, dose, strength, concentration,
     drawn/withdrawn amount) = Meaning-changed.
   - If 5 mL (or any volume) is specified in the Entity and omitted in candidate
     → Parent at best, NEVER Equivalent.
   - adenocarcinoma ≠ carcinoma.
     carcinoma is Parent of adenocarcinoma.
     carcinoma MUST NOT be treated as Equivalent to adenocarcinoma.

1. FIRST: Classify EVERY candidate explicitly as one of:
   - fully Equivalent with entity
   - Parent (Broader) of entity
   - Child (Narrower) of entity
   - Meaning-changed with entity

   Do NOT discard any candidate.
   Classification MUST precede scoring.

2. If ONE OR MORE Equivalent candidates exist:
   - The best Equivalent MUST receive the highest score (5.0).
   - ALL Parent concepts MUST score strictly lower than the Equivalent.
   - Parent concepts become invalid for final selection in this scenario.
   - Selecting a Parent when an Equivalent exists is a hierarchy violation.

3. If NO Equivalent exists:
   - Evaluate valid Parent concepts.
   - Parent must fully contain the entity meaning without contradiction.
   - Parent may omit attributes but MUST NOT alter subtype or quantitative values.

4. Child or Meaning-changed concepts:
   - MUST receive ≤ 1.5.
   - They are never valid mappings.

Entity: {entity_name}

Candidates:
{candidates_json}


### Output Requirements
- Every candidate MUST receive a score.
- Every candidate MUST have a UNIQUE score using decimal precision (e.g., 4.8, 5.0, 2.3, 0.0).
- Output valid JSON only.

Output Format (JSON only):
{{
  "rankings": [
    {{
        "concept_id": "ID",
        "reasoning": "[Step-by-step logic addressing whether it is Equivalent, Parent, or Child, explicitly mentioning added/missing info]",
        "score": 0.0-5.0}},
    ...
  ]
}}
"""


class Stage3LLMScoring:
    """Stage 3: LLM-based scoring and ranking of Standard Concept candidates."""

    # Prompt templates (can be overridden by subclasses)
    SYSTEM_PROMPT = SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE = USER_PROMPT_TEMPLATE

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize Stage 3.

        Args:
            llm_client: Pre-configured LLM client (a new one is created if None)
            llm_provider: LLM route key (openai, together)
            llm_model: Optional model override
            llm_base_url: Optional OpenAI-compatible base URL override
            llm_api_key: Optional API key override
            temperature: LLM temperature (default from env or 0.3)
            top_p: LLM top_p / nucleus sampling (default from env or 1.0)
            max_tokens: LLM max output tokens (default from env or model default)
        """
        self.llm_client = llm_client
        self.temperature = temperature
        self.top_p = top_p

        if self.llm_client is None:
            self.llm_client = get_llm_client(
                provider=llm_provider,
                model=llm_model,
                base_url=llm_base_url,
                api_key=llm_api_key,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )

        if self.llm_client.is_initialized:
            llm_info = self.llm_client.get_info()
            logger.info(
                f"Stage 3 initialized (LLM scoring, "
                f"provider: {llm_info['provider']}, model: {llm_info['model']})"
            )
        else:
            logger.error("LLM client not initialized")

    def score_candidates(
        self,
        entity_name: str,
        stage2_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Score Stage 2 candidates with the LLM and rank them.

        Args:
            entity_name: Source term to map
            stage2_candidates: Standard Concept candidates from Stage 2

        Returns:
            Candidates sorted by LLM score (descending); empty list on failure
        """
        if not stage2_candidates:
            logger.warning("No candidates to score")
            return []

        if not self.llm_client or not self.llm_client.is_initialized:
            logger.error("LLM client not initialized")
            return []

        results = []
        for candidate in stage2_candidates:
            es_score = candidate.get('elasticsearch_score', 0.0)
            results.append({
                'concept': candidate['concept'],
                'is_original_standard': candidate.get('is_original_standard', True),
                'original_candidate': candidate.get('original_candidate', {}),
                'original_non_standard': candidate.get('original_non_standard'),
                'relation_type': candidate.get('relation_type', 'original'),
                'elasticsearch_score': es_score,
                'elasticsearch_score_normalized': sigmoid_normalize(es_score, center=3.0),
                'search_type': candidate.get('search_type', 'unknown')
            })

        try:
            llm_scores = self._call_llm(entity_name, results)

            if not llm_scores:
                logger.error("LLM scoring failed")
                return []

            for candidate in results:
                cid = str(candidate['concept'].get('concept_id', ''))
                if cid in llm_scores:
                    candidate['llm_score'] = llm_scores[cid]['score']
                    candidate['llm_rank'] = llm_scores[cid]['rank']
                    candidate['llm_reasoning'] = llm_scores[cid].get('reasoning', '')
                    candidate['final_score'] = candidate['llm_score']
                else:
                    candidate['llm_score'] = 0.0
                    candidate['llm_rank'] = 999
                    candidate['final_score'] = 0.0

            sorted_results = sorted(results, key=lambda x: x['llm_score'], reverse=True)

            self._log_results(sorted_results)
            return sorted_results

        except Exception as e:
            logger.error(f"LLM scoring failed: {e}")
            return []

    def _call_llm(
        self,
        entity_name: str,
        candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Send the scoring prompt to the LLM and parse the rankings."""
        prompt = self._build_prompt(entity_name, candidates)

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            response = self.llm_client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                json_mode=True,
                metrics_tag="stage3_llm_scoring",
            )

            if response is None:
                return None

            return self._parse_llm_response(response, candidates)

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return None

    def _build_prompt(self, entity_name: str, candidates: List[Dict[str, Any]]) -> str:
        """
        Build the user prompt.

        Each candidate carries index, concept_id and concept_name. Candidates
        reached via "Maps to" also carry the original non-standard concept.
        """
        candidates_info = []
        for i, c in enumerate(candidates, 1):
            concept = c['concept']
            info = {
                'index': i,
                'concept_id': str(concept.get('concept_id', '')),
                'concept_name': concept.get('concept_name', ''),
            }

            original_non_std = c.get('original_non_standard')
            if c.get('relation_type') == 'Maps to' and original_non_std:
                info['original_concept'] = {
                    'concept_id': str(original_non_std.get('concept_id', '')),
                    'concept_name': original_non_std.get('concept_name', ''),
                }

            candidates_info.append(info)

        return self.USER_PROMPT_TEMPLATE.format(
            entity_name=entity_name,
            candidates_json=json.dumps(candidates_info, ensure_ascii=False, indent=2),
        )

    def _parse_llm_response(
        self,
        response_text: str,
        candidates: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Parse the LLM JSON response into {concept_id: {score, rank, reasoning}}."""
        try:
            text = response_text.strip()
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            parsed = json.loads(text)
            result = {}

            rankings = parsed.get('rankings', [])

            # Sort by score descending and assign ranks
            rankings_sorted = sorted(rankings, key=lambda x: float(x.get('score', 0)), reverse=True)

            for rank, item in enumerate(rankings_sorted, 1):
                cid = str(item.get('concept_id', ''))
                if cid:
                    result[cid] = {
                        'score': float(item.get('score', 0.0)),
                        'rank': rank,
                        'reasoning': item.get('reasoning', '')
                    }

            # Candidates the LLM did not rank get the lowest score
            for c in candidates:
                cid = str(c['concept'].get('concept_id', ''))
                if cid not in result:
                    result[cid] = {'score': 0.0, 'rank': 999, 'reasoning': 'Not ranked'}

            return result

        except Exception as e:
            logger.error(f"LLM response parsing failed: {e}")
            return {}

    def _log_results(self, results: List[Dict[str, Any]]):
        """Log top results as: concept_name (concept_id) score, then reasoning."""
        logger.info("Stage 3: Scoring Results")
        for r in results[:15]:
            concept = r['concept']
            name = concept.get('concept_name', 'N/A')
            cid = concept.get('concept_id', 'N/A')
            score = r.get('final_score', 0.0)
            logger.info(f"  {name} ({cid}) {score:.4f}")
            reasoning = r.get('llm_reasoning', '')
            if reasoning:
                logger.info(f"    → {reasoning}")
