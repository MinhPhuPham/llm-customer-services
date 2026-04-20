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
# multilingual-e5-small: XLM-R architecture (12L/384H), 117M params.
# Contrastive pretraining (2024) gives strong EN+JP embeddings.
# Standard BERT ops → clean TFLite/CoreML export.
BASE_MODEL = 'intfloat/multilingual-e5-small'


# ===========================================================
# LANGUAGE PREFIXES
# ===========================================================
# E5 models require 'query: ' prefix for input texts.
LANG_PREFIX = {'en': 'query: [EN]', 'ja': 'query: [JA]'}


# ===========================================================
# TRAINING PARAMETERS
# ===========================================================
MAX_SEQ_LENGTH = 64
CONFIDENCE_THRESHOLD = 0.40


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
    """Detect GPU and return (device, dtype, gpu_info, is_high_end)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        device = 'cuda'
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        # High-end GPUs: A100, H100, A10, etc. (support bf16 and have >30GB)
        is_high_end = torch.cuda.is_bf16_supported() and gpu_mem > 30
        return device, dtype, f"{gpu_name} ({gpu_mem:.1f}GB)", is_high_end
    return 'cpu', torch.float32, None, False


DEVICE, DTYPE, GPU_INFO, IS_HIGH_END_GPU = detect_device()


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
elif IS_HIGH_END_GPU:
    # Large-batch recipe for A100/H100 (Colab/Kaggle).
    # Sqrt LR scaling rule (Hoffer et al. 2017):
    #   LR_new = LR_base × sqrt(batch_new / batch_base)
    #   1.6e-4 = 2e-5  × sqrt(2048 / 32)
    # 30 epochs ensures enough optimizer steps even for small datasets
    # (e.g. 55 samples → 1 step/epoch → 30 total steps).
    NUM_EPOCHS = 30
    BATCH_SIZE = 2048
    LEARNING_RATE = 1.6e-4
else:
    # Standard recipe for T4/V100/consumer GPUs or CPU.
    # Smaller batch + standard LR for stable training.
    # More epochs to compensate for smaller batch updates.
    NUM_EPOCHS = 30
    BATCH_SIZE = 32 if DEVICE == 'cuda' else 16
    LEARNING_RATE = 2e-5


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
    gpu_mode = "Large-batch (A100)" if IS_HIGH_END_GPU else "Standard-batch"
    print("=" * 60)
    print(f"  Platform:    {PLATFORM}")
    print(f"  Mode:        {mode}")
    print(f"  Device:      {DEVICE} ({GPU_INFO or 'CPU — will be slow'})")
    print(f"  GPU mode:    {gpu_mode}")
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Epochs:      {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"  Max seq len: {MAX_SEQ_LENGTH}")
    print(f"  Confidence:  {CONFIDENCE_THRESHOLD}")
    print(f"  Export dir:  {EXPORT_DIR}")
    print("=" * 60)
