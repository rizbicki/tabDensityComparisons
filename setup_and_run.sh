#!/usr/bin/env bash
set -euo pipefail

# FlexCode × Tabular Foundation Models: Preliminary CDE Experiments
# =================================================================
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh              # full run (8 datasets)
#   ./setup_and_run.sh --quick      # quick sanity check (4 datasets)
#   ./setup_and_run.sh --cpu        # force CPU
#
# This script:
#   1. Creates a Python venv in .venv/
#   2. Installs all dependencies (TabPFN, TabICLv2, XGBoost, etc.)
#   3. Runs the experiments
#   4. Outputs results to results/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

# Parse args — pass everything through to the Python script
EXTRA_ARGS=()
DEVICE="auto"
for arg in "$@"; do
    case "$arg" in
        --cpu) DEVICE="cpu"; EXTRA_ARGS+=("--device" "cpu") ;;
        --quick) EXTRA_ARGS+=("--quick") ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

# ── 1. Create venv ──────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in ${VENV_DIR}/..."
    python3 -m venv "$VENV_DIR"
    echo "✓ venv created"
else
    echo "✓ venv already exists at ${VENV_DIR}/"
fi

# ── 2. Upgrade pip ──────────────────────────────────────────────
echo "Upgrading pip..."
"$PIP" install --upgrade pip --quiet

# ── 3. Install dependencies ─────────────────────────────────────
echo "Installing dependencies..."
# numpy must be installed first — xbart's setup.py imports it at build time
"$PIP" install --quiet numpy
"$PIP" install --quiet -r requirements.txt

# Check what actually installed
echo ""
echo "Installed models:"
"$PYTHON" -c "
try:
    import tabpfn; print('  ✓ TabPFN', tabpfn.__version__ if hasattr(tabpfn, '__version__') else '')
except ImportError:
    print('  ✗ TabPFN — install failed, will skip')
try:
    import tabicl; print('  ✓ TabICLv2')
except ImportError:
    print('  ✗ TabICLv2 — install failed, will skip')
try:
    import xgboost; print('  ✓ XGBoost', xgboost.__version__)
except ImportError:
    print('  ✗ XGBoost — will use sklearn GBM')
try:
    import xbart; print('  ✓ XBART')
except ImportError:
    print('  ✗ XBART — install failed, will skip BART methods')
try:
    import torch; print('  ✓ PyTorch', torch.__version__, '(CUDA:', torch.cuda.is_available(), ')')
except ImportError:
    print('  ✗ PyTorch not found')
"

# ── 4. Run experiments ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Running experiments..."
echo "============================================================"
echo ""

"$PYTHON" run_experiments.py "${EXTRA_ARGS[@]}"

echo ""
echo "============================================================"
echo "  Done! Results are in results/"
echo "============================================================"
