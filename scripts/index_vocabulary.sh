#!/bin/bash
# OMOP vocabulary indexing script
#
# ── Basic usage ──
#   ./scripts/index_vocabulary.sh                           # Default settings (concept-small/relationship/synonym)
#   ./scripts/index_vocabulary.sh --prepare-only            # Generate CONCEPT_SMALL.csv only
#
# ── Specifying tables (multiple allowed) ──
#   ./scripts/index_vocabulary.sh --tables concept-small synonym
#   ./scripts/index_vocabulary.sh --tables concept-small    # concept-small only
#
# ── Add only 'Is a' relationships to existing concept-relationship (no deletion of existing data) ──
#   ./scripts/index_vocabulary.sh --add-isa
#
# ── Restart after an interruption (Checkpoint-based) ──
#   ./scripts/index_vocabulary.sh --resume                  # Read the last successful position from the checkpoint and resume
#   ./scripts/index_vocabulary.sh --resume --tables synonym
#
# ── Mitigate 429s (wait between bulk requests) ──
#   ./scripts/index_vocabulary.sh --resume --bulk-delay 1
#
# ── Test (partial rows only) ──
#   ./scripts/index_vocabulary.sh --max-rows 10000
#
# ── Specifying the vocabulary folder ──
#   DATA_FOLDER=/path/to/vocabulary ./scripts/index_vocabulary.sh
#   ./scripts/index_vocabulary.sh --data-folder /path/to/vocabulary
#
# ── Safety guarantees ──
#   - Idempotent _id: re-sending the same data overwrites (no duplicates)
#   - Checkpoint: record progress per chunk, restart from that chunk on failure
#   - 429 backoff: exponential backoff retry of 5-300s (up to 7 times)
#   - Individual failures: automatically retry failed documents within a bulk response (up to 3 times)
#   - Verification: after completion, compare ES document count vs source row count

set -e

# Find the project root relative to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# ============================================================================
# Settings (edit if needed)
# ============================================================================
DATA_FOLDER="${DATA_FOLDER:-$PROJECT_ROOT/data/omop-cdm}"

# ============================================================================
# Main logic
# ============================================================================

echo "============================================"
echo "OMOP vocabulary indexing"
echo "============================================"
echo "Project: $PROJECT_ROOT"
echo "Data folder: $DATA_FOLDER"
echo "============================================"

# Check the --prepare-only option
if [[ "$1" == "--prepare-only" ]]; then
    echo ""
    echo "[Step] Generate CONCEPT_SMALL.csv"
    echo "--------------------------------------------"
    python scripts/create_concept_small.py --data-folder "$DATA_FOLDER"
    echo ""
    echo "Done!"
    exit 0
fi

# --add-isa: CONCEPT_SMALL not needed (uses CONCEPT_RELATIONSHIP only)
ADD_ISA=false
for arg in "$@"; do [[ "$arg" == "--add-isa" ]] && ADD_ISA=true && break; done

# Generate CONCEPT_SMALL.csv if missing (skipped in --add-isa mode)
if [[ "$ADD_ISA" != "true" ]]; then
    CONCEPT_SMALL_PATH="$DATA_FOLDER/CONCEPT_SMALL.csv"

    echo ""
    echo "[Step 1/2] Check CONCEPT_SMALL.csv"
    echo "--------------------------------------------"

    if [[ -f "$CONCEPT_SMALL_PATH" ]]; then
        echo "  -> Already exists: $CONCEPT_SMALL_PATH"
        echo "  -> To regenerate: ./scripts/index_vocabulary.sh --prepare-only"
    else
        echo "  -> Creating..."
        python scripts/create_concept_small.py --data-folder "$DATA_FOLDER"
        echo "  -> Done"
    fi

    echo ""
    echo "[Step 2/2] Elasticsearch indexing"
    echo "--------------------------------------------"
else
    echo ""
    echo "[Step] Add 'Is a' relationships to concept-relationship"
    echo "--------------------------------------------"
fi

# Run indexing (pass DATA_FOLDER to Python; a command-line --data-folder takes precedence)
python scripts/index_vocabulary.py --data-folder "$DATA_FOLDER" "$@"

echo ""
echo "============================================"
echo "Done!"
echo "============================================"
