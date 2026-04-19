# ===========================================================
# export_coreml.py — PyTorch → CoreML .mlpackage (FP16)
# ===========================================================
"""
Exports the trained PyTorch classifier to CoreML format
for on-device iOS inference via Apple Neural Engine.

Uses torch.export + core_aten_decompositions to ensure all ops
decompose to primitives that coremltools can handle.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

from scripts.config import MAX_SEQ_LENGTH, EXPORT_DIR


class _FlatWrapper(nn.Module):
    """Wraps HF classifier to return a plain logits tensor (no dict)."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return out.logits


def export_coreml(trainer=None, tokenizer=None, export_dir=None):
    """
    Export model to CoreML .mlpackage (FP16).

    Loads the saved model from best_pytorch/ directory.
    The trainer parameter is kept for backward compatibility but unused.

    Returns:
        coreml_path: Path to saved .mlpackage.
        coreml_size_mb: Size in MB.
    """
    export_dir = export_dir or EXPORT_DIR
    best_dir = os.path.join(export_dir, 'best_pytorch')

    print("=" * 60)
    print("EXPORT: CoreML (iOS)")
    print("=" * 60)

    from transformers import AutoModelForSequenceClassification

    if not os.path.exists(best_dir):
        raise FileNotFoundError(
            f"Model not found at {best_dir}. "
            "Run save_best_model (Step 8b) first."
        )

    print("  Loading saved model...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(best_dir)
    wrapper = _FlatWrapper(hf_model).eval().cpu().float()

    dummy_ids = torch.randint(0, 100, (1, MAX_SEQ_LENGTH), dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.long)

    print("  [1/3] torch.export...")
    exported = torch.export.export(
        wrapper,
        (dummy_ids, dummy_mask),
        strict=False,
    )

    print("  [2/3] Decomposing to ATEN primitives...")
    from torch._decomp import core_aten_decompositions
    exported = exported.run_decompositions(core_aten_decompositions())

    print("  [3/3] CoreML conversion (FP16 mlprogram)...")
    mlmodel = ct.convert(
        exported,
        inputs=[
            ct.TensorType(
                name='input_ids',
                shape=(1, MAX_SEQ_LENGTH),
                dtype=np.int32,
            ),
            ct.TensorType(
                name='attention_mask',
                shape=(1, MAX_SEQ_LENGTH),
                dtype=np.int32,
            ),
        ],
        outputs=[ct.TensorType(name='logits')],
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.author = 'SupportAI'
    mlmodel.short_description = 'Bilingual EN/JP tag classifier'
    mlmodel.version = '1.0'

    # Save
    coreml_path = os.path.join(export_dir, 'SupportAI.mlpackage')
    mlmodel.save(coreml_path)

    coreml_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(coreml_path)
        for f in fn
    ) / 1e6

    print(f"  CoreML: {coreml_size:.1f}MB (FP16)")
    return coreml_path, coreml_size
