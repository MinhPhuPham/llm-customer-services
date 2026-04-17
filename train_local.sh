#!/usr/bin/env bash
# ===========================================================
# train_local.sh — Local venv setup + dependency validation
# ===========================================================
#
# Creates an isolated .venv, installs all pipeline dependencies,
# and validates that the full chain works (imports, ONNX export,
# onnx2tf → TFLite conversion).
#
# Usage:
#   ./train_local.sh              # Setup + validate
#   ./train_local.sh setup        # Just create venv + install
#   ./train_local.sh validate     # Just run validation checks
#   ./train_local.sh clean        # Remove venv
#
# Everything stays inside this project directory — no global installs.
# ===========================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

do_setup() {
    header "Creating virtual environment"

    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}  .venv already exists — reusing. Run '$0 clean' to start fresh.${NC}"
    else
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}  Created $VENV_DIR${NC}"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel -q

    header "Installing dependencies"

    # ── 1. PyTorch (CPU — local validation, GPU training on Colab) ──
    echo "  [1/7] PyTorch (CPU)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q 2>&1 | tail -1

    # ── 2. Core ML deps (install first — they set numpy/protobuf baseline) ──
    echo "  [2/7] Transformers + training..."
    pip install "transformers>=4.51.0" accelerate -q 2>&1 | tail -1
    pip install datasets scikit-learn openpyxl -q 2>&1 | tail -1
    pip install matplotlib tqdm -q 2>&1 | tail -1

    # ── 3. TensorFlow (for TFLite evaluation + optional INT8 converter) ──
    echo "  [3/7] TensorFlow..."
    pip install tensorflow -q 2>&1 | tail -1

    # ── 4. ONNX + onnx2tf (TFLite export) ──
    # --no-deps on onnx2tf avoids pulling onnxsim (C++ build fails on macOS)
    # and its strict protobuf==4.25.5 / numpy==1.26.4 pins
    echo "  [4/7] ONNX + onnx2tf..."
    pip install onnx onnxruntime onnxscript -q 2>&1 | tail -1
    pip install onnx2tf --no-deps -q 2>&1 | tail -1
    pip install ai-edge-litert flatbuffers -q 2>&1 | tail -1
    pip install sng4onnx sne4onnx --no-deps -q 2>&1 | tail -1 || true

    # ── 5. CoreML export (macOS native) ──
    echo "  [5/7] CoreML tools..."
    pip install coremltools zstandard -q 2>&1 | tail -1

    # ── 6. Google Sheets support ──
    echo "  [6/7] Google Sheets..."
    pip install gspread google-auth -q 2>&1 | tail -1

    echo "  [7/7] Done."

    echo ""
    echo -e "${GREEN}  All dependencies installed${NC}"
}

do_validate() {
    if [ ! -d "$VENV_DIR" ]; then
        echo -e "${RED}  No .venv found. Run '$0 setup' first.${NC}"
        exit 1
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    # ── Check 1: Library versions ──
    header "Library versions"

    python3 << 'PYEOF'
import sys

libs = [
    ("torch",          lambda: __import__("torch").__version__),
    ("numpy",          lambda: __import__("numpy").__version__),
    ("protobuf",       lambda: __import__("google.protobuf", fromlist=["__version__"]).__version__),
    ("transformers",   lambda: __import__("transformers").__version__),
    ("onnx",           lambda: __import__("onnx").__version__),
    ("onnx2tf",        lambda: getattr(__import__("onnx2tf"), "__version__", "installed")),
    ("onnxruntime",    lambda: __import__("onnxruntime").__version__),
    ("onnxscript",     lambda: __import__("onnxscript").__version__),
    ("tensorflow",     lambda: __import__("tensorflow").__version__),
    ("coremltools",    lambda: __import__("coremltools").__version__),
    ("scikit-learn",   lambda: __import__("sklearn").__version__),
    ("datasets",       lambda: __import__("datasets").__version__),
    ("flatbuffers",    lambda: __import__("flatbuffers").__version__),
]

ok, fail = 0, 0
for name, get_ver in libs:
    try:
        ver = get_ver()
        print(f"  \033[0;32mOK\033[0m   {name:20s} {ver}")
        ok += 1
    except Exception as e:
        print(f"  \033[0;31mFAIL\033[0m {name:20s} {e}")
        fail += 1

print(f"\n  {ok} OK, {fail} failed")
if fail:
    sys.exit(1)
PYEOF

    # ── Check 2: Pipeline module imports ──
    header "Pipeline module imports"

    python3 << PYEOF
import sys
sys.path.insert(0, "$PROJECT_DIR")

modules = [
    "scripts.config",
    "scripts.setup",
    "scripts.data_loader.excel_parser",
    "scripts.data_loader.dataset_builder",
    "scripts.data_loader.export_artifacts",
    "scripts.helpers.vocab_pruner",
    "scripts.helpers.trainer_factory",
    "scripts.helpers.training_plots",
    "scripts.helpers.export_coreml",
    "scripts.helpers.export_tflite",
    "scripts.helpers.compress",
    "scripts.helpers.evaluator",
]

ok, fail = 0, 0
for mod in modules:
    try:
        __import__(mod)
        print(f"  \033[0;32mOK\033[0m   {mod}")
        ok += 1
    except Exception as e:
        print(f"  \033[0;31mFAIL\033[0m {mod}: {type(e).__name__}: {e}")
        fail += 1

print(f"\n  {ok} OK, {fail} failed")
if fail:
    sys.exit(1)
PYEOF

    # ── Check 3: ONNX → TFLite export chain ──
    header "TFLite export chain (dummy model)"

    python3 << 'PYEOF'
import torch, torch.nn as nn, tempfile, os, glob, sys

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)
    def forward(self, x):
        return self.linear(x)

model = Dummy().eval()
x = torch.randn(1, 10)

with tempfile.TemporaryDirectory() as tmp:
    # Step 1: PyTorch → ONNX
    onnx_path = os.path.join(tmp, "test.onnx")
    with torch.no_grad():
        torch.onnx.export(
            model, x, onnx_path, opset_version=17,
            input_names=["input"], output_names=["output"],
        )
    sz = os.path.getsize(onnx_path)
    print(f"  [1/3] PyTorch → ONNX:     \033[0;32mOK\033[0m ({sz:,} bytes)")

    # Step 2: ONNX → onnx2tf
    out_dir = os.path.join(tmp, "out")
    import onnx2tf
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        non_verbose=True,
        copy_onnx_input_output_names_to_tflite=True,
    )

    contents = os.listdir(out_dir)
    tflite_files = glob.glob(os.path.join(out_dir, "*.tflite"))
    has_saved_model = os.path.exists(os.path.join(out_dir, "saved_model.pb"))

    print(f"  [2/3] ONNX → onnx2tf:     \033[0;32mOK\033[0m")
    print(f"         Output files: {contents}")
    print(f"         SavedModel:   {has_saved_model}")
    print(f"         TFLite files: {len(tflite_files)}")

    # Step 3: Try INT8 via TFLiteConverter
    if has_saved_model:
        try:
            import tensorflow as tf
            converter = tf.lite.TFLiteConverter.from_saved_model(out_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_bytes = converter.convert()
            print(f"  [3/3] INT8 quantization:  \033[0;32mOK\033[0m ({len(tflite_bytes):,} bytes)")
        except Exception as e:
            print(f"  [3/3] INT8 quantization:  \033[0;33mSKIP\033[0m ({e})")
            if tflite_files:
                sz = os.path.getsize(tflite_files[0])
                print(f"        FP32 fallback:      \033[0;32mOK\033[0m ({sz:,} bytes)")
    elif tflite_files:
        sz = os.path.getsize(tflite_files[0])
        print(f"  [3/3] Flatbuffer direct:  \033[0;32mOK\033[0m ({sz:,} bytes)")
    else:
        print(f"  [3/3] \033[0;31mFAIL\033[0m No TFLite output!")
        sys.exit(1)

print(f"\n  \033[0;32mTFLite export chain: VALIDATED\033[0m")
PYEOF

    # ── Check 4: numpy binary compatibility ──
    header "numpy binary compatibility"

    python3 << 'PYEOF'
import sys

checks = [
    ("numpy.random (mtrand)", lambda: __import__("numpy.random")),
    ("numpy.linalg", lambda: __import__("numpy.linalg")),
    ("torch + numpy", lambda: __import__("torch").randn(2, 2).numpy()),
    ("sklearn + numpy", lambda: __import__("sklearn.metrics")),
    ("tensorflow + numpy", lambda: __import__("tensorflow")),
]

ok, fail = 0, 0
for name, check in checks:
    try:
        check()
        print(f"  \033[0;32mOK\033[0m   {name}")
        ok += 1
    except Exception as e:
        print(f"  \033[0;31mFAIL\033[0m {name}: {type(e).__name__}: {e}")
        fail += 1

print(f"\n  {ok} OK, {fail} failed")
if fail:
    print("  numpy binary incompatibility detected!")
    sys.exit(1)
PYEOF

    header "ALL CHECKS PASSED"
    echo ""
    echo "  Resolved versions work together. Use these in Colab:"
    echo ""

    python3 << 'PYEOF'
libs = ["torch", "numpy", "protobuf", "onnx", "onnx2tf",
        "onnxruntime", "onnxscript", "tensorflow", "flatbuffers"]
for lib in libs:
    try:
        if lib == "protobuf":
            ver = __import__("google.protobuf", fromlist=["__version__"]).__version__
        elif lib == "onnx2tf":
            ver = getattr(__import__("onnx2tf"), "__version__", "?")
        else:
            ver = __import__(lib).__version__
        print(f"  {lib}=={ver}")
    except Exception:
        pass
PYEOF
}

do_clean() {
    header "Cleaning up"
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        echo -e "${GREEN}  Removed $VENV_DIR${NC}"
    else
        echo "  No .venv found"
    fi
}

# ── Main ──
CMD="${1:-setup}"

case "$CMD" in
    setup)
        do_setup
        do_validate
        ;;
    validate)
        do_validate
        ;;
    clean)
        do_clean
        ;;
    *)
        echo "Usage: $0 [setup|validate|clean]"
        exit 1
        ;;
esac
