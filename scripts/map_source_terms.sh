#!/bin/bash
# Batch mapping: source terms -> OMOP Standard Concepts
#
# Selecting a dataset applies its default CSV path and preprocessing automatically
# (see DATA_SOURCES in scripts/mapping_io.py).
# Output: test_logs/mapping_{snuh|snomed}_{timestamp}.{json,log,xlsx}
#
# ============================================================================
# Options
# ============================================================================
#
# [Dataset] required
#   snuh   : data/snuh-baseline-mapping-data.csv
#   snomed : data/snomed-mapping-data-1000.csv
#
# [Sampling]
#   -n, --sample-size N   : Max number of samples (all data if omitted). Ignored when --sample-per-domain is used
#   --sample-per-domain N : Sample N per domain. Example: --sample-per-domain 5
#   --random              : Random sampling
#   --seed N              : Random seed (default: 42)
#
# [LLM route selection]
#   --llm-provider   : openai | together
#   --llm-model      : Model name override (Together: gpt_oss_20b | mistral_small_24b | llama4_maverick aliases supported)
#   --llm-base-url   : OpenAI-compatible endpoint override
#   --llm-api-key-env: Name of the environment variable to read the API key from
#   --llm-temperature: temperature override
#   --llm-top-p      : top_p override
#   --llm-max-tokens : Max output tokens override
#
# [Parallel processing]
#   -w, --workers N  : Number of worker processes (default: 1). 4-8 recommended (~1GB memory per worker)
#
# [Repeat mapping] (consistency check)
#   -r, --repeat N   : Map the same data N times (default: 1). 5 generates a summary + 5 detail sheets.
#
# ============================================================================
# Examples
# ============================================================================
#
# SNUH (full data):
#   ./scripts/map_source_terms.sh snuh
#
# SNUH, 5 per domain, random:
#   ./scripts/map_source_terms.sh snuh --sample-per-domain 5 --random
#
# SNOMED (full data):
#   ./scripts/map_source_terms.sh snomed
#
# Parallel, 4 workers (recommended for 1000+ items):
#   ./scripts/map_source_terms.sh snuh --workers 4
#
# Repeat 5 times (consistency check, summary + 5 detail sheets):
#   ./scripts/map_source_terms.sh snuh --repeat 5
#
# Together GPT-OSS-20B:
#   ./scripts/map_source_terms.sh snuh --llm-provider together --llm-model gpt_oss_20b --llm-api-key-env TOGETHER_API_KEY
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Real-time log output (disable buffering)
export PYTHONUNBUFFERED=1

echo "============================================"
echo "MapOMOP batch mapping"
echo "============================================"
echo "Project: $PROJECT_ROOT"
echo "============================================"

python scripts/map_source_terms.py "$@"

echo ""
echo "============================================"
echo "Done! Check .json, .log, .xlsx in test_logs/"
echo "============================================"
