# Data Samples

Small, real samples that show what goes **into** and comes **out of** each part of
MapOMOP. Full vocabulary files and evaluation datasets are not distributed with this
repository.

```text
samples/
├── vocabulary/                  # 1) OMOP vocabulary files (Athena download format, tab-separated)
│   ├── CONCEPT.csv              #    10 rows
│   ├── CONCEPT_SYNONYM.csv      #    10 rows
│   ├── CONCEPT_RELATIONSHIP.csv #    10 rows
│   └── CONCEPT_SMALL.csv        #    20 rows = CONCEPT (10) + CONCEPT_SYNONYM (10), built by create_concept_small.py
├── index/                       # 2) The same rows as Elasticsearch documents (_index, _id, _source)
│   ├── concept-small.json       #    20 docs (10 Original + 10 Synonym), with 128-dim SapBERT embeddings
│   ├── concept-synonym.json     #    10 docs
│   └── concept-relationship.json#    10 docs
└── mapping/                     # 3) Mapping CLI input and output
    ├── source_terms.csv         #    10 source terms with ground-truth concepts
    └── mapping_result.json      #    Mapping result for the 10 terms
```

## 1. Vocabulary → 2. Index

The sample covers 10 concepts: 6 Standard Concepts (SNOMED, LOINC) and 4 non-standard
concepts (ICD10CM, ICD9CM, ICD9Proc, and one deprecated SNOMED concept) that reach a
Standard Concept through `Maps to`.

`index/` documents were exported from the Elasticsearch cluster used by the demo.
Running the indexing code on `vocabulary/` reproduces them: all fields are
identical, and embeddings differ only by floating-point noise (GPU fp16 vs. CPU fp32).

### CONCEPT → `concept-small`

Each CONCEPT row becomes one `Original` document. Concept names are lowercased, dates
are normalized to `YYYYMMDD`, empty values become `null`, and a SapBERT embedding
(768 → 128 dims, L2-normalized) is added. The document `_id` is
`md5("{concept_id}_{concept_name}")`, so re-indexing overwrites rather than duplicates.

Raw row (`vocabulary/CONCEPT.csv`):

| concept_id | concept_name | domain_id | vocabulary_id | concept_class_id | standard_concept | concept_code | valid_start_date | valid_end_date | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4186397 | Myocardial ischemia | Condition | SNOMED | Disorder | S | 414795007 | 20050131 | 20991231 | |
| 40436953 | Myocardial ischemia | Condition | SNOMED | Disorder | | 233822007 | 20020131 | 20020131 | D |

Indexed document (`index/concept-small.json`, embedding truncated):

```json
{
  "_index": "concept-small",
  "_id": "ed69a241bb0518c7d9a27f8c39cf849c",
  "_source": {
    "concept_id": "4186397",
    "concept_name": "myocardial ischemia",
    "name_type": "Original",
    "domain_id": "Condition",
    "vocabulary_id": "SNOMED",
    "concept_class_id": "Disorder",
    "standard_concept": "S",
    "concept_code": "414795007",
    "valid_start_date": "20050131",
    "valid_end_date": "20991231",
    "invalid_reason": null,
    "concept_embedding": [-0.0267, -0.2156, 0.0397, "... 128 values"]
  }
}
```

### CONCEPT_SYNONYM → `concept-small` (Synonym) and `concept-synonym`

English synonyms (`language_concept_id = 4180186`) are added to `concept-small` as
`Synonym` documents that inherit the concept's metadata, so a search on a synonym can
be resolved back to its concept in Stage 1. The raw table is also indexed as-is into
`concept-synonym`.

| CONCEPT_SYNONYM row | `concept-small` document |
| --- | --- |
| `4186397 · Cardiac ischemia · 4180186` | `concept_name: "cardiac ischemia"`, `name_type: "Synonym"`, `standard_concept: "S"`, ... |
| `40436953 · Myocardial ischaemia · 4180186` | `concept_name: "myocardial ischaemia"`, `name_type: "Synonym"`, `standard_concept: null`, ... |

### CONCEPT_RELATIONSHIP → `concept-relationship`

Only the relationships used by Stage 2 are indexed (`Maps to`, `Is a`,
`Concept alt_to to`, `Concept poss_eq to`, `Concept same_as to`, `Marketed form of`,
`Tradename of`, `Box of`, `Has quantified form`). The document `_id` is
`md5("{concept_id_1}_{concept_id_2}_{relationship_id}")`.

| concept_id_1 | concept_id_2 | relationship_id | Meaning |
| --- | --- | --- | --- |
| 1567956 (ICD10CM E11) | 201826 (SNOMED) | Maps to | Non-standard → Standard Concept |
| 40436953 (SNOMED, deprecated) | 4186397 (SNOMED) | Maps to | Deprecated → current Standard Concept |
| 4186397 | 4186397 | Maps to | Standard Concepts map to themselves |
| 4186397 | 4185932 | Is a | Parent concept |

## 3. Mapping input → output

`mapping/source_terms.csv` uses the column layout of the SNUH dataset read by
`scripts/map_source_terms.py snuh` (`source_value` is the term to map, `concept_id` is
the ground truth):

| no | domain_id | source_value | concept_id | concept_name |
| --- | --- | --- | --- | --- |
| 310 | Condition | Atrial fibrillation | 313217 | atrial fibrillation |
| 718 | Drug | Lamotrigine 100mg tab | 705108 | lamotrigine 100 mg oral tablet |
| 2820 | Measurement | Blood Urea Nitrogen | 3013682 | urea nitrogen [mass/volume] in serum or plasma |
| 928 | Procedure | Closed [endoscopic] biopsy of stomach | 4004241 | endoscopic biopsy of stomach |

`mapping/mapping_result.json` holds the mapping records for these terms, taken from
run 1 of the 20-run evaluation. Each record has the final mapping plus the candidates
kept at each stage:

| Field | Description |
| --- | --- |
| `entity_name`, `input_domain` | Source term and target domain |
| `ground_truth_concept_id`, `ground_truth_concept_name` | Expected Standard Concept |
| `best_concept_id`, `best_concept_name`, `best_score` | Selected Standard Concept and its LLM score (0–5) |
| `mapping_correct` | `best_concept_id == ground_truth_concept_id` |
| `stage1_candidates` | Retrieved concepts with `search_type` (`lexical` / `semantic` / `combined`) and normalized `elasticsearch_score` |
| `stage2_candidates` | Standard Concepts with `relation_type` (`original`, `Maps to`, `Is a`, ...) and `original_non_standard` when converted |
| `stage3_candidates` | Standard Concepts with `llm_score`, `llm_rank`, and `llm_reasoning` |

Example: `Blood Urea Nitrogen` (Measurement), where the selected concept differs from
the ground truth (abridged; only the rank-2 Stage 3 candidate is shown).

```json
{
  "entity_name": "Blood Urea Nitrogen",
  "input_domain": "Measurement",
  "ground_truth_concept_id": 3013682,
  "ground_truth_concept_name": "urea nitrogen [mass/volume] in serum or plasma",
  "success": true,
  "mapping_correct": false,
  "best_concept_id": "4017361",
  "best_concept_name": "blood urea nitrogen measurement",
  "best_score": 5.0,
  "stage3_candidates": [
    {
      "concept_id": "4094594",
      "concept_name": "blood urea measurement",
      "is_original_standard": false,
      "original_non_standard": { "concept_id": "45616467", "concept_name": "blood urea nitrogen" },
      "llm_score": 4.8,
      "llm_rank": 2,
      "llm_reasoning": "Classification: Equivalent (mapped-from evidence). ..."
    }
  ]
}
```

## Licensing note

Vocabulary rows are excerpts of the OMOP Standardized Vocabularies from
[OHDSI Athena](https://athena.ohdsi.org). Each source vocabulary keeps its own license
(for example, SNOMED CT and LOINC). Download the full vocabularies from Athena under
those terms.
