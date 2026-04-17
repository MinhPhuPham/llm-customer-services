# ===========================================================
# config.py — All constants, paths, hyperparams, modes
# ===========================================================
"""
Central configuration for the bilingual customer support AI pipeline.
Switch between TESTING and PRODUCTION mode here.
"""

import os
import torch


# ===========================================================
# MODE
# ===========================================================
TESTING_MODE = False
CONTINUE_TRAINING = False


# ===========================================================
# BASE MODEL
# ===========================================================
# mmBERT-small recommended (ModernBERT arch, 2025)
# Fallback: 'xlm-roberta-base'
BASE_MODEL = 'jhu-clsp/mmbert-small'


# ===========================================================
# LANGUAGE PREFIXES
# ===========================================================
LANG_PREFIX = {'en': '[EN]', 'ja': '[JA]'}


# ===========================================================
# TRAINING PARAMETERS
# ===========================================================
MAX_SEQ_LENGTH = 64
CONFIDENCE_THRESHOLD = 0.85


# ===========================================================
# PLATFORM DETECTION
# ===========================================================
def detect_platform():
    """Detect runtime platform: colab / kaggle / local."""
    if os.path.exists('/kaggle/working'):
        return 'kaggle'
    elif os.path.exists('/content'):
        return 'colab'
    return 'local'


PLATFORM = detect_platform()


# ===========================================================
# GPU / DEVICE
# ===========================================================
def detect_device():
    """Detect GPU and return (device, dtype, gpu_info)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        device = 'cuda'
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return device, dtype, f"{gpu_name} ({gpu_mem:.1f}GB)"
    return 'cpu', torch.float32, None


DEVICE, DTYPE, GPU_INFO = detect_device()


# ===========================================================
# DIRECTORY PATHS
# ===========================================================
def build_paths(platform):
    """Build directory paths based on platform."""
    if platform == 'colab':
        drive_dir = '/content/drive/MyDrive/SupportAI'
    elif platform == 'kaggle':
        drive_dir = '/kaggle/working/output'
    else:
        # Local mode — store outputs alongside the project
        drive_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'output'
        )

    return {
        'drive_dir': drive_dir,
        'data_dir': os.path.join(drive_dir, 'data'),
        'model_dir': os.path.join(drive_dir, 'models'),
        'export_dir': os.path.join(drive_dir, 'export'),
    }


PATHS = build_paths(PLATFORM)
DATA_DIR = PATHS['data_dir']
MODEL_DIR = PATHS['model_dir']
EXPORT_DIR = PATHS['export_dir']


# ===========================================================
# TRAINING HYPERPARAMS (mode-dependent)
# ===========================================================
if TESTING_MODE:
    NUM_EPOCHS = 3
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
else:
    # Large-batch recipe for A100 (Colab/Kaggle).
    # Sqrt LR scaling rule (Hoffer et al. 2017):
    #   LR_new = LR_base × sqrt(batch_new / batch_base)
    #   1.6e-4 = 2e-5  × sqrt(2048 / 32)
    # 30 epochs ensures enough optimizer steps even for small datasets
    # (e.g. 55 samples → 1 step/epoch → 30 total steps).
    NUM_EPOCHS = 15
    BATCH_SIZE = 2048 if DEVICE == 'cuda' else 16
    LEARNING_RATE = 1.6e-4


# ===========================================================
# REPO CONFIG (for Colab/Kaggle — clone private GitHub repo)
# ===========================================================
REPO_BRANCH = 'main'
REPO_OWNER = '<YourGitHubUser>'
REPO_NAME = '<YourRepoName>'


# ===========================================================
# PRINT SUMMARY
# ===========================================================
def print_config():
    """Print current configuration summary."""
    mode = "TESTING" if TESTING_MODE else "PRODUCTION"
    print("=" * 60)
    print(f"  Platform:    {PLATFORM}")
    print(f"  Mode:        {mode}")
    print(f"  Device:      {DEVICE} ({GPU_INFO or 'CPU — will be slow'})")
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Epochs:      {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"  Max seq len: {MAX_SEQ_LENGTH}")
    print(f"  Confidence:  {CONFIDENCE_THRESHOLD}")
    print(f"  Export dir:  {EXPORT_DIR}")
    print("=" * 60)
