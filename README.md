# MapOMOP

MapOMOP maps free-text clinical terms (conditions, drugs, measurements, procedures,
observations) to **OMOP CDM Standard Concepts**. It combines Elasticsearch retrieval,
SapBERT semantic embeddings, OMOP concept relationships, and LLM scoring, and ships
with a Streamlit demo and command-line tools.

A hosted demo is available at **https://mapomop.onrender.com**.

## Method

Given a source term and an optional target domain, mapping runs as a three-stage
pipeline (`src/MapOMOP/mapping_stages/`):

1. **Candidate retrieval** (`stage1_candidate_retrieval.py`)
   Retrieves candidate concepts from the `concept-small` index with three complementary
   strategies: lexical search (exact / phrase / fuzzy), semantic vector search over
   SapBERT embeddings, and a combined query that mixes text, vector, and length
   similarity. Synonym hits are resolved back to their original concepts.

2. **Standard concept collection** (`stage2_standard_concept_collection.py`)
   Converts candidates to OMOP Standard Concepts (`standard_concept` = `S` or `C`) by
   following `CONCEPT_RELATIONSHIP` links (e.g. `Is a`, `Tradename of`) and `Maps to`,
   in two rounds.

3. **LLM scoring** (`stage3_llm_scoring.py`)
   An LLM scores every Standard Concept candidate (0–5) using OMOP hierarchy rules:
   an equivalent concept is preferred, a parent concept is allowed only when no
   equivalent exists, and child or meaning-changed concepts are rejected. The top
   candidate is the final mapping. LLM access is provider-agnostic via `LLMClient`
   (OpenAI and Together AI).

## Requirements

- Python 3.10+ (3.11 recommended)
- An OpenAI API key (or another supported LLM provider)
- Access to the OMOP Elasticsearch indexes (host, port, credentials)

## Setup

```bash
git clone https://github.com/rose2gyqls/MapOMOP.git
cd MapOMOP

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in the values
```

Minimum `.env` values:

```bash
OPENAI_API_KEY=your-openai-api-key
ES_SERVER_HOST=your-es-host
ES_SERVER_PORT=9200
ES_SERVER_USERNAME=your-es-username
ES_SERVER_PASSWORD=your-es-password
ES_USE_SSL=false
```

## Usage

### Streamlit demo

```bash
streamlit run scripts/app.py
```

Enter a clinical term and a target domain to see the best Standard Concept and the
ranked candidates.

### Python API

```python
from MapOMOP import EntityInput, EntityMappingAPI, DomainID

api = EntityMappingAPI()
entity = EntityInput(entity_name="myocardial ischemia", domain_id=DomainID.CONDITION)
results = api.map_entity(entity)
```

### Mapping CLI

Runs batch mapping over a dataset registered in `scripts/mapping_io.py` and writes
`.json`, `.log`, and `.xlsx` to `test_logs/`.

```bash
./scripts/map_source_terms.sh snuh      # or: snomed
```

Common options:

| Option | Description |
| --- | --- |
| `-n, --sample-size N` | Limit the number of samples |
| `--sample-per-domain N` | Sample N terms per domain |
| `--random`, `--seed N` | Random sampling and seed |
| `-w, --workers N` | Parallel worker processes |
| `-r, --repeat N` | Repeat mapping N times (consistency check) |
| `--llm-provider {openai,together}`, `--llm-model` | LLM route override |

### Indexing CLI

Builds the Elasticsearch indexes from OMOP vocabulary files downloaded from
[Athena](https://athena.ohdsi.org). Only needed if you maintain your own indexes; the
demo and mapping CLI just need access to an existing cluster.

```bash
./scripts/index_vocabulary.sh                                   # vocabulary folder: data/omop-cdm
./scripts/index_vocabulary.sh --data-folder /path/to/vocabulary
```

| Index | Source | Used by |
| --- | --- | --- |
| `concept-small` | `CONCEPT` + English `CONCEPT_SYNONYM` (built by `create_concept_small.py`), with SapBERT embeddings | Stage 1, Stage 2 |
| `concept-relationship` | `CONCEPT_RELATIONSHIP` (relationships used by Stage 2 only) | Stage 2 |
| `concept-synonym` | `CONCEPT_SYNONYM` | Synonym lookup (`ElasticsearchClient.search_synonyms`) |

## Data Samples

[`samples/`](./samples) contains small real examples of every data format, with a
[walkthrough](./samples/README.md):

| Folder | Contents |
| --- | --- |
| [`samples/vocabulary/`](./samples/vocabulary) | Athena vocabulary rows: `CONCEPT`, `CONCEPT_SYNONYM`, `CONCEPT_RELATIONSHIP` (10 each) and the derived `CONCEPT_SMALL` |
| [`samples/index/`](./samples/index) | The same rows as indexed Elasticsearch documents |
| [`samples/mapping/`](./samples/mapping) | 10 source terms and their mapping results (Stage 1–3 candidates, LLM reasoning) |

For example, one `CONCEPT` row becomes one `concept-small` document:

```text
CONCEPT.csv   4186397 | Myocardial ischemia | Condition | SNOMED | Disorder | S | 414795007 | 20050131 | 20991231 |
concept-small {"concept_id": "4186397", "concept_name": "myocardial ischemia", "name_type": "Original",
               "standard_concept": "S", ..., "concept_embedding": [-0.0267, -0.2156, 0.0397, ... 128 dims]}
```

## Deployment

The app is deployed on [Render](https://render.com) and served at
**https://mapomop.onrender.com**. A [`render.yaml`](./render.yaml) Blueprint is
included: push the repository, create a Render Blueprint from it, and provide the
secrets (`OPENAI_API_KEY`, `ES_SERVER_HOST`, `ES_SERVER_USERNAME`,
`ES_SERVER_PASSWORD`). The start command is:

```bash
streamlit run scripts/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM scoring |
| `OPENAI_MODEL` | No | OpenAI model override (default `gpt-5-mini-2025-08-07`) |
| `ES_SERVER_HOST` | Yes | Elasticsearch host |
| `ES_SERVER_PORT` | No | Elasticsearch port (default `9200`) |
| `ES_SERVER_USERNAME` | Yes | Elasticsearch username |
| `ES_SERVER_PASSWORD` | Yes | Elasticsearch password |
| `ES_USE_SSL` | No | `true` or `false` (default `false`) |

Never commit `.env`. Keep Elasticsearch credentials out of tracked source files and
prefer read-only credentials for demo users.

## Project Structure

```text
MapOMOP/
├── src/MapOMOP/                            # Core mapping package
│   ├── entity_mapping_api.py               # EntityMappingAPI (pipeline entry point)
│   ├── mapping_stages/
│   │   ├── stage1_candidate_retrieval.py
│   │   ├── stage2_standard_concept_collection.py
│   │   └── stage3_llm_scoring.py
│   ├── elasticsearch_client.py
│   ├── llm_client.py
│   └── utils.py                            # Embedding projection, deduplication
├── indexing/                               # Elasticsearch index-building pipeline
│   ├── data_sources/
│   │   ├── base.py
│   │   └── read_vocabulary.py              # Athena vocabulary CSV reader
│   ├── vocabulary_indexer.py               # Orchestrates indexing per table
│   ├── elasticsearch_indexer.py
│   └── sapbert_embedder.py
├── scripts/                                # CLIs and wrappers
│   ├── app.py                              # Streamlit demo
│   ├── map_source_terms.py                 # Mapping CLI   (map_source_terms.sh)
│   ├── mapping_io.py                       # Dataset loading, logging, JSON/XLSX output
│   ├── index_vocabulary.py                 # Indexing CLI  (index_vocabulary.sh)
│   └── create_concept_small.py             # CONCEPT_SMALL.csv builder
├── samples/                                # Data samples (see samples/README.md)
├── requirements.txt
├── render.yaml                             # Render deployment blueprint
└── .env.example
```

## License

MIT
