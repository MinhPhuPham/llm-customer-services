# ===========================================================
# export_tflite.py — PyTorch → TFLite .tflite (INT8)
# ===========================================================
"""
Exports the trained model to TFLite format with INT8 quantization
for Android (NNAPI/GPU delegate).

When calibration data is provided, uses full INT8 quantization
(both weights and activations) for best on-device performance.
Otherwise falls back to INT8 dynamic range quantization.

Tries TFLITE_BUILTINS-only first to avoid the ~20MB TF Flex
delegate; falls back to SELECT_TF_OPS if needed.
"""

import os

import numpy as np

from scripts.config import MAX_SEQ_LENGTH, EXPORT_DIR


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

    print("=" * 60)
    print("EXPORT: TFLite (Android)")
    print("=" * 60)

    # Lazy import — tensorflow only loaded when export is called
    import tensorflow as tf
    from transformers import TFAutoModelForSequenceClassification

    # Load as TF model from PyTorch weights
    tf_model = TFAutoModelForSequenceClassification.from_pretrained(
        best_dir, from_pt=True
    )

    # Concrete function with fixed input shape
    @tf.function(input_signature=[
        tf.TensorSpec([1, MAX_SEQ_LENGTH], tf.int32, name='input_ids'),
        tf.TensorSpec([1, MAX_SEQ_LENGTH], tf.int32, name='attention_mask'),
    ])
    def serve(input_ids, attention_mask):
        output = tf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return {'logits': output.logits}

    # Build representative dataset for full INT8 quantization
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

        print(f"  Calibration: {len(cal_samples)} samples for full INT8")
    else:
        print("  No calibration data — using dynamic range quantization")

    # Convert: try TFLITE_BUILTINS only first (avoids ~20MB Flex delegate),
    # fall back to SELECT_TF_OPS if the model uses unsupported ops.
    concrete_fn = serve.get_concrete_function()
    tflite_model = None

    ops_configs = [
        ('TFLITE_BUILTINS', [tf.lite.OpsSet.TFLITE_BUILTINS]),
        ('TFLITE_BUILTINS + SELECT_TF_OPS', [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]),
    ]

    for ops_label, ops in ops_configs:
        try:
            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [concrete_fn]
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_fn is not None:
                converter.representative_dataset = representative_fn
            converter.target_spec.supported_ops = ops
            tflite_model = converter.convert()
            print(f"  Ops: {ops_label}")
            break
        except Exception as e:
            if ops_label == ops_configs[-1][0]:
                raise
            print(f"  {ops_label} failed, retrying with SELECT_TF_OPS...")

    # Save
    tflite_path = os.path.join(export_dir, 'support_ai.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    tflite_size = os.path.getsize(tflite_path) / 1e6
    quant_type = 'full INT8' if representative_fn else 'INT8 dynamic range'
    print(f"  TFLite: {tflite_size:.1f}MB ({quant_type})")

    return tflite_path, tflite_size
