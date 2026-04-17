# ===========================================================
# export_tflite.py — PyTorch → TFLite via ai-edge-torch
# ===========================================================
"""
Exports the trained model to TFLite for Android (NNAPI/GPU delegate).

Uses ai-edge-torch — Google's official 2024 PyTorch → TFLite converter.
Skips the dead onnx/onnx-tf chain entirely (StableHLO intermediate).

Key trick: set `return_dict=False` on the HF model so it returns
a plain tuple instead of SequenceClassifierOutput dict.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from scripts.config import MAX_SEQ_LENGTH, EXPORT_DIR


class _FlatWrapper(nn.Module):
    """Wraps HF classifier to return a plain logits tensor (no dict)."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
        self.model.config.return_dict = False

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return out[0]  # logits


def export_tflite(tokenizer=None, calibration_texts=None, export_dir=None):
    """
    Export model to TFLite .tflite via ai-edge-torch.

    Args:
        tokenizer: Unused (kept for API compatibility).
        calibration_texts: Unused (kept for API compatibility).
        export_dir: Override export directory.

    Returns:
        tflite_path: Path to saved .tflite file.
        tflite_size_mb: Size in MB.
    """
    export_dir = export_dir or EXPORT_DIR
    best_dir = os.path.join(export_dir, 'best_pytorch')

    print("=" * 60)
    print("EXPORT: TFLite (Android)")
    print("=" * 60)

    # Lazy imports — heavy deps only needed at export time
    import ai_edge_torch
    from transformers import AutoModelForSequenceClassification

    # Load saved PyTorch model and wrap
    print("  Loading saved model...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(best_dir)
    wrapper = _FlatWrapper(hf_model).eval().cpu().float()

    dummy_ids = torch.randint(0, 100, (1, MAX_SEQ_LENGTH), dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.long)
    sample_inputs = (dummy_ids, dummy_mask)

    # Apply dynamic INT8 quantization (no calibration needed, ~4x smaller)
    print("  Converting PyTorch → TFLite (dynamic INT8)...")
    try:
        from ai_edge_torch.quantize.pt2e_quantizer import (
            PT2EQuantizer, get_symmetric_quantization_config,
        )
        from ai_edge_torch.quantize.quant_config import QuantConfig

        quantizer = PT2EQuantizer().set_global(
            get_symmetric_quantization_config(
                is_per_channel=True, is_dynamic=True,
            )
        )
        edge_model = ai_edge_torch.convert(
            wrapper, sample_inputs,
            quant_config=QuantConfig(pt2e_quantizer=quantizer),
        )
        quant_label = 'dynamic INT8'
    except Exception as e:
        print(f"  Quantization failed ({type(e).__name__}: {e}), falling back to FP32")
        edge_model = ai_edge_torch.convert(wrapper, sample_inputs)
        quant_label = 'FP32'

    # Save
    tflite_path = os.path.join(export_dir, 'support_ai.tflite')
    edge_model.export(tflite_path)

    tflite_size = os.path.getsize(tflite_path) / 1e6
    print(f"  TFLite: {tflite_size:.1f}MB ({quant_label})")

    return tflite_path, tflite_size
