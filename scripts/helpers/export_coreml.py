# ===========================================================
# export_coreml.py — PyTorch → CoreML .mlpackage (FP16)
# ===========================================================
"""
Exports the trained PyTorch classifier to CoreML format
for on-device iOS inference via Apple Neural Engine.

Key trick: set `return_dict=False` on the HF model so it returns
a plain tuple instead of SequenceClassifierOutput. This eliminates
the `dictconstruct` op that coremltools can't convert.
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
        # Force tuple output — kills the SequenceClassifierOutput dict ops
        self.model.config.return_dict = False

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return out[0]  # logits


def export_coreml(trainer, tokenizer, export_dir=None):
    """
    Export model to CoreML .mlpackage (FP16).

    Args:
        trainer: HF Trainer with trained model.
        tokenizer: HuggingFace tokenizer.
        export_dir: Override export directory.

    Returns:
        coreml_path: Path to saved .mlpackage.
        coreml_size_mb: Size in MB.
    """
    export_dir = export_dir or EXPORT_DIR

    print("=" * 60)
    print("EXPORT: CoreML (iOS)")
    print("=" * 60)

    # Save best PyTorch model first
    best_dir = os.path.join(export_dir, 'best_pytorch')
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    print(f"  Saved PyTorch model to {best_dir}")

    # Prepare wrapped model for tracing (float32 on CPU)
    wrapper = _FlatWrapper(trainer.model).eval().cpu().float()

    dummy_ids = torch.randint(0, 100, (1, MAX_SEQ_LENGTH), dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.long)

    # TorchScript trace — no dict construction inside now
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_ids, dummy_mask), strict=False)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
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
    mlmodel.short_description = 'Bilingual EN/JP intent classifier'
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
