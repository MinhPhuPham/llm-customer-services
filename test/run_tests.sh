#!/usr/bin/env bash
# ===========================================================
# run_tests.sh — Run test suite for bilingual support AI
# ===========================================================
#
# Usage:
#   ./test/run_tests.sh                # Run all tests
#   ./test/run_tests.sh data           # Data pipeline tests only
#   ./test/run_tests.sh tokenize       # Tokenization tests only
#   ./test/run_tests.sh inference      # Inference tests only (needs model)
#   ./test/run_tests.sh conversation   # Conversation flow tests (needs model)
#   ./test/run_tests.sh greeting       # Greeting-related tests
#   ./test/run_tests.sh quick          # Data + tokenization (no model needed)
#   ./test/run_tests.sh chat           # Interactive chat (test model like a user)
#   ./test/run_tests.sh chat ja        # Interactive chat in Japanese
#
# Prerequisites:
#   pip install pytest transformers datasets scikit-learn openpyxl numpy
#   For inference/conversation/chat: trained + exported TFLite model
# ===========================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Use venv if it exists
if [ -d "$PROJECT_ROOT/.venv" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

FILTER="${1:-all}"

case "$FILTER" in
    chat)
        LANG="${2:-en}"
        exec python test/chat.py --lang "$LANG"
        ;;
    *)
        echo "============================================================"
        echo "  Bilingual Support AI — Test Suite"
        echo "============================================================"
        echo "  Project: $PROJECT_ROOT"
        echo ""
        ;;
esac

if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${RED}ERROR: pytest not installed. Run: pip install pytest${NC}"
    exit 1
fi

EXCEL_FILE="$PROJECT_ROOT/service_model_training_data_template.xlsx"
if [ ! -f "$EXCEL_FILE" ]; then
    echo -e "${YELLOW}WARNING: Excel training data not found.${NC}"
    echo "  Expected: $EXCEL_FILE"
    echo "  Data pipeline tests will be skipped."
    echo ""
fi

case "$FILTER" in
    data)
        echo -e "${GREEN}Running: Data pipeline tests${NC}"
        python -m pytest test/test_pipeline.py -v -k "TestDataPipeline" --tb=short
        ;;
    tokenize|token)
        echo -e "${GREEN}Running: Tokenization tests${NC}"
        python -m pytest test/test_pipeline.py -v -k "TestTokenization" --tb=short
        ;;
    inference|infer)
        echo -e "${GREEN}Running: Inference tests (requires trained model)${NC}"
        python -m pytest test/test_pipeline.py -v -k "TestInference" --tb=short
        ;;
    conversation|conv|flow)
        echo -e "${GREEN}Running: Conversation flow tests (requires trained model)${NC}"
        python -m pytest test/test_pipeline.py -v -k "TestConversationFlow" --tb=short
        ;;
    greeting)
        echo -e "${GREEN}Running: Greeting-related tests${NC}"
        python -m pytest test/test_pipeline.py -v -k "greeting" --tb=short
        ;;
    quick)
        echo -e "${GREEN}Running: Quick tests (data + tokenization, no model needed)${NC}"
        python -m pytest test/test_pipeline.py -v -k "TestDataPipeline or TestTokenization" --tb=short
        ;;
    all)
        echo -e "${GREEN}Running: All tests${NC}"
        python -m pytest test/test_pipeline.py -v --tb=short
        ;;
    *)
        echo "Unknown filter: $FILTER"
        echo ""
        echo "Usage: $0 [data|tokenize|inference|conversation|greeting|quick|chat|all]"
        echo ""
        echo "  chat [en|ja]   Interactive chat — test model like a real user"
        echo "  data           Data pipeline tests (no model needed)"
        echo "  tokenize       Tokenization tests"
        echo "  inference      Inference tests (needs exported model)"
        echo "  conversation   Conversation flow tests (needs exported model)"
        echo "  greeting       Greeting-related tests"
        echo "  quick          Data + tokenization (no model needed)"
        echo "  all            All tests"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  Tests complete"
echo "============================================================"
