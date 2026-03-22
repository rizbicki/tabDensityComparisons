#!/usr/bin/env bash
set -euo pipefail

# FlexCode × Tabular Foundation Models: Preliminary CDE Experiments
# =================================================================
# Usage:
#   chmod +x setup_and_run.sh
#   ./setup_and_run.sh --setup-only     # create .venv and install dependencies only
#   ./setup_and_run.sh --sim-only       # simulated datasets only
#   ./setup_and_run.sh                  # full run (synthetic + real)
#   ./setup_and_run.sh --real-only      # real/semi-synthetic datasets only
#   ./setup_and_run.sh --cpu            # force CPU
#
# This script:
#   1. Creates a Python venv in .venv/
#   2. Installs all dependencies (TabPFN, TabICLv2, XGBoost, etc.)
#   3. Runs the experiments
#   4. Outputs simulated results to results_simulated/ and real results to results_real/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

run_with_device_env() {
    if [ "$DEVICE" = "cpu" ]; then
        CUDA_VISIBLE_DEVICES="" "$@"
    else
        "$@"
    fi
}

# Parse args — pass everything through to the Python script
EXTRA_ARGS=()
DEVICE="auto"
REAL_ONLY=0
SIM_ONLY=0
SETUP_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --cpu) DEVICE="cpu"; EXTRA_ARGS+=("--device" "cpu") ;;
        --sim-only) SIM_ONLY=1; EXTRA_ARGS+=("--sim-only") ;;
        --real-only) REAL_ONLY=1 ;;
        --setup-only) SETUP_ONLY=1 ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

if [ "$REAL_ONLY" -eq 1 ] && [ "$SIM_ONLY" -eq 1 ]; then
    echo "Cannot combine --real-only and --sim-only."
    exit 1
fi

if [ "$DEVICE" = "cpu" ]; then
    echo "CPU mode requested; hiding CUDA devices during startup and runs."
fi

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
# numpy + setuptools must be installed first — xbart's setup.py imports numpy at build time
"$PIP" install --quiet numpy setuptools
# xbart needs --no-build-isolation since its setup.py imports numpy directly
"$PIP" install --quiet --no-build-isolation xbart
"$PIP" install --quiet -r requirements.txt

# Check what actually installed
echo ""
echo "Installed models:"
run_with_device_env "$PYTHON" -c "
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
    import torch
    if '${DEVICE}' == 'cpu':
        print('  ✓ PyTorch', torch.__version__, '(CUDA: hidden for CPU mode)')
    else:
        print('  ✓ PyTorch', torch.__version__, '(CUDA:', torch.cuda.is_available(), ')')
except ImportError:
    print('  ✗ PyTorch not found')
"

if [ "$SETUP_ONLY" -eq 1 ]; then
    echo ""
    echo "============================================================"
    echo "  Setup complete. Virtual environment is ready at ${VENV_DIR}/"
    echo "============================================================"
    exit 0
fi

# ── 4. Run experiments ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Running experiments..."
echo "============================================================"
echo ""

if [ "$REAL_ONLY" -eq 1 ]; then
    run_with_device_env "$PYTHON" run_real_experiments.py --n-reps 4 "${EXTRA_ARGS[@]}"
    echo ""
    echo "============================================================"
    echo "  Done! Results are in results_real/"
    echo "============================================================"
else
    run_with_device_env "$PYTHON" run_experiments.py --n-reps 4 "${EXTRA_ARGS[@]}"
    echo ""
    echo "============================================================"
    echo "  Done! Simulated results are in results_simulated/"
    if [ "$SIM_ONLY" -eq 0 ]; then
        echo "  Done! Real results are in results_real/"
    fi
    echo "============================================================"
fi
