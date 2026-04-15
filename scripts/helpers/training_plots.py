# ===========================================================
# training_plots.py — Loss / accuracy / F1 charts
# ===========================================================
"""
Generates training history plots:
  - Train vs Val loss
  - Validation accuracy
  - Validation F1 (macro)
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from scripts.config import EXPORT_DIR


def plot_training_history(trainer, export_dir=None, show=True):
    """
    Generate and save training history plots.

    Args:
        trainer: HF Trainer (after training).
        export_dir: Override export directory.
        show: If True, display plot inline (for notebooks).

    Returns:
        best_accuracy: Best validation accuracy achieved.
    """
    export_dir = export_dir or EXPORT_DIR

    log_history = trainer.state.log_history
    train_loss = [
        x['loss'] for x in log_history
        if 'loss' in x and 'eval_loss' not in x
    ]
    eval_entries = [x for x in log_history if 'eval_loss' in x]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss ---
    axes[0].plot(train_loss, alpha=0.5, label='Train')
    axes[0].plot(
        np.linspace(0, len(train_loss), len(eval_entries)),
        [x['eval_loss'] for x in eval_entries],
        'r-o', label='Val',
    )
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot([x['eval_accuracy'] for x in eval_entries], 'g-o')
    axes[1].set_title('Val Accuracy')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    # --- F1 ---
    axes[2].plot([x['eval_f1_macro'] for x in eval_entries], 'b-o')
    axes[2].set_title('Val F1 (macro)')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(export_dir, 'training_plots.png')
    plt.savefig(plot_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


    best_accuracy = max(x['eval_accuracy'] for x in eval_entries)
    print(f"  Saved: training_plots.png")
    print(f"  Best accuracy: {best_accuracy:.4f}")

    return best_accuracy
