# Lazy __init__ — no eager imports of heavy deps (tensorflow, coremltools).
# Import directly from each module:
#   from scripts.helpers.vocab_pruner import prune_vocabulary
#   from scripts.helpers.export_tflite import export_tflite
#   etc.

__all__ = [
    'vocab_pruner',
    'trainer_factory',
    'training_plots',
    'export_coreml',
    'export_tflite',
    'compress',
    'evaluator',
]
