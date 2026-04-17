# ===========================================================
# export_tflite.py — PyTorch → ONNX → TFLite (INT8)
# ===========================================================
"""
Exports the trained model to TFLite format with INT8 quantization
for Android (NNAPI/GPU delegate).

Pipeline: PyTorch → ONNX → TF SavedModel → TFLite (INT8)

When calibration data is provided, uses full INT8 quantization
(both weights and activations) for best on-device performance.
"""

import os

import numpy as np
import torch
import torch.nn as nn

from scripts.config import MAX_SEQ_LENGTH, EXPORT_DIR


class _LogitsWrapper(nn.Module):
    """Wraps a HF classifier to return only the logits tensor.

    HuggingFace models return a dict-like SequenceClassifierOutput.
    ONNX export needs a clean tensor output for reliable conversion.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export_tflite(tokenizer=None, calibration_texts=None, export_dir=None):
    """
    Export model to TFLite .tflite (INT8 quantized).

    Args:
        tokenizer: HuggingFace tokenizer (or RemappedTokenizer) for calibration.
                   If provided with calibration_texts, enables full INT8.
        calibration_texts: List of text samples for representative dataset.
        export_dir: Override export directory.

    Returns:
        tflite_path: Path to saved .tflite file.
        tflite_size_mb: Size in MB.
    """
    export_dir = export_dir or EXPORT_DIR
    best_dir = os.path.join(export_dir, 'best_pytorch')
    onnx_path = os.path.join(export_dir, 'model.onnx')
    saved_model_dir = os.path.join(export_dir, 'tf_saved_model')

    print("=" * 60)
    print("EXPORT: TFLite (Android)")
    print("=" * 60)

    # -------------------------------------------------------
    # Step 1: PyTorch → ONNX
    # -------------------------------------------------------
    from transformers import AutoModelForSequenceClassification

    print("  [1/3] PyTorch → ONNX...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(best_dir)
    wrapper = _LogitsWrapper(hf_model).eval().cpu().float()

    dummy_ids = torch.randint(0, 100, (1, MAX_SEQ_LENGTH), dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes=None,
            opset_version=14,
            do_constant_folding=True,
        )
    print(f"        Saved ONNX: {os.path.getsize(onnx_path) / 1e6:.1f}MB")

    # -------------------------------------------------------
    # Step 2: ONNX → TF SavedModel
    # -------------------------------------------------------
    print("  [2/3] ONNX → TF SavedModel...")
    import onnx
    from onnx_tf.backend import prepare

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(saved_model_dir)
    print(f"        Saved TF model to {saved_model_dir}")

    # -------------------------------------------------------
    # Step 3: TF SavedModel → TFLite (INT8)
    # -------------------------------------------------------
    print("  [3/3] TF SavedModel → TFLite (INT8)...")
    import tensorflow as tf

    representative_fn = None
    if tokenizer is not None and calibration_texts is not None:
        cal_samples = calibration_texts[:200]

        def representative_fn():
            for text in cal_samples:
                enc = tokenizer(
                    text, return_tensors='np',
                    padding='max_length', truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                )
                yield [
                    enc['input_ids'].astype(np.int32),
                    enc['attention_mask'].astype(np.int32),
                ]

        print(f"        Calibration: {len(cal_samples)} samples for full INT8")
    else:
        print("        No calibration data — using dynamic range quantization")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if representative_fn is not None:
        converter.representative_dataset = representative_fn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int32
        converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    # Save
    tflite_path = os.path.join(export_dir, 'support_ai.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    tflite_size = os.path.getsize(tflite_path) / 1e6
    quant_type = 'full INT8' if representative_fn else 'INT8 dynamic range'
    print(f"  TFLite: {tflite_size:.1f}MB ({quant_type})")

    # Cleanup intermediate files
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

    return tflite_path, tflite_size
