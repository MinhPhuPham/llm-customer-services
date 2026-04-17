# ===========================================================
# export_tflite.py — PyTorch → TFLite via ONNX + onnx2tf
# ===========================================================
"""
Exports the trained model to TFLite for Android (NNAPI/GPU delegate).

Pipeline: PyTorch → ONNX (opset 17) → TF SavedModel (onnx2tf) → TFLite.
Uses TensorFlow's native TFLiteConverter for dynamic range INT8 quantization.
"""

import os
import tempfile

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
    Export model to TFLite via ONNX → onnx2tf → TFLiteConverter.

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

    from transformers import AutoModelForSequenceClassification

    print("  Loading saved model...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(best_dir)
    wrapper = _FlatWrapper(hf_model).eval().cpu().float()

    dummy_ids = torch.randint(0, 100, (1, MAX_SEQ_LENGTH), dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.long)

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, 'model.onnx')

        # Step 1: PyTorch → ONNX
        print("  [1/3] PyTorch → ONNX (opset 17)...")
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_ids, dummy_mask),
                onnx_path,
                input_names=['input_ids', 'attention_mask'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch'},
                    'attention_mask': {0: 'batch'},
                    'logits': {0: 'batch'},
                },
                opset_version=17,
            )

        # Step 2: ONNX → TF SavedModel
        print("  [2/3] ONNX → TF SavedModel (onnx2tf)...")
        import onnx2tf
        saved_model_dir = os.path.join(tmp_dir, 'tf_saved_model')
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            non_verbose=True,
            copy_onnx_input_output_names_to_tflite=True,
        )

        # Step 3: TF SavedModel → TFLite (dynamic range INT8)
        print("  [3/3] SavedModel → TFLite (dynamic INT8)...")
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        try:
            tflite_model = converter.convert()
            quant_label = 'dynamic INT8'
        except Exception as e:
            print(f"  INT8 failed ({type(e).__name__}: {e}), falling back to FP32")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            tflite_model = converter.convert()
            quant_label = 'FP32'

    # Save
    tflite_path = os.path.join(export_dir, 'support_ai.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    tflite_size = os.path.getsize(tflite_path) / 1e6
    print(f"  TFLite: {tflite_size:.1f}MB ({quant_label})")

    return tflite_path, tflite_size
