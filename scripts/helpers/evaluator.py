# ===========================================================
# evaluator.py — TFLite inference + bilingual tests + latency
# ===========================================================
"""
Evaluates the exported TFLite model:
  - Validation accuracy (TFLite interpreter)
  - Bilingual test cases (EN + JA)
  - Inference latency measurement
"""

import time

import numpy as np

from scripts.config import MAX_SEQ_LENGTH, LANG_PREFIX, CONFIDENCE_THRESHOLD


class TFLiteEvaluator:
    """Evaluates a TFLite model with bilingual test cases."""

    def __init__(self, tflite_path, tokenizer, label_map):
        """
        Args:
            tflite_path: Path to .tflite model file.
            tokenizer: HuggingFace tokenizer.
            label_map: Dict[int, str] mapping label ID → intent name.
        """
        self.tokenizer = tokenizer
        self.label_map = label_map

        # Lazy import — tensorflow only loaded when evaluator is used
        import tensorflow as tf

        # Load interpreter
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.inp_details = self.interpreter.get_input_details()
        self.out_details = self.interpreter.get_output_details()

        # Map input names → indices (onnx2tf may reorder inputs)
        self._input_index = {}
        for detail in self.inp_details:
            name = detail['name'].lower()
            if 'input_id' in name:
                self._input_index['input_ids'] = detail['index']
            elif 'attention' in name or 'mask' in name:
                self._input_index['attention_mask'] = detail['index']
        # Fallback: assume positional order if names don't match
        if 'input_ids' not in self._input_index:
            self._input_index['input_ids'] = self.inp_details[0]['index']
            self._input_index['attention_mask'] = self.inp_details[1]['index']

    def predict(self, text, lang='en', threshold=None):
        """
        Run inference on a single text.

        Args:
            text: Raw query text (without language prefix).
            lang: 'en' or 'ja'.
            threshold: Confidence threshold (default: config value).

        Returns:
            (intent, confidence): Predicted intent and confidence score.
        """
        threshold = threshold if threshold is not None else CONFIDENCE_THRESHOLD

        full_text = f"{LANG_PREFIX[lang]} {text}"
        enc = self.tokenizer(
            full_text,
            return_tensors='np',
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

        self.interpreter.set_tensor(
            self._input_index['input_ids'],
            enc['input_ids'].astype(np.int32),
        )
        self.interpreter.set_tensor(
            self._input_index['attention_mask'],
            enc['attention_mask'].astype(np.int32),
        )
        self.interpreter.invoke()

        logits = self.interpreter.get_tensor(self.out_details[0]['index'])
        logits_stable = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits_stable) / np.exp(logits_stable).sum(axis=-1, keepdims=True)
        pred_id = np.argmax(probs, axis=-1)[0]
        confidence = probs[0][pred_id]
        intent = self.label_map[int(pred_id)]

        if confidence < threshold:
            return 'out_of_scope', confidence
        return intent, confidence

    def run_validation(self, val_texts, val_labels, label_encoder):
        """
        Run validation accuracy on the full validation set.

        Args:
            val_texts: List of prefixed validation texts.
            val_labels: Encoded validation labels.
            label_encoder: Fitted LabelEncoder.

        Returns:
            accuracy: Float accuracy score.
        """
        correct = 0
        for text, label in zip(val_texts, val_labels):
            # Strip prefix for predict()
            clean = text.replace('[EN] ', '').replace('[JA] ', '')
            lang = 'en' if text.startswith('[EN]') else 'ja'
            pred_intent, _ = self.predict(clean, lang=lang, threshold=0.0)

            if label_encoder.classes_[label] == pred_intent:
                correct += 1

        accuracy = correct / len(val_texts) * 100
        return accuracy


def evaluate_model(tflite_path, tokenizer, label_map, label_encoder,
                   val_texts, val_labels):
    """
    Full evaluation: validation accuracy + bilingual tests + latency.

    Args:
        tflite_path: Path to .tflite model.
        tokenizer: HuggingFace tokenizer.
        label_map: Dict[int, str] label mapping.
        label_encoder: Fitted LabelEncoder.
        val_texts: Validation texts.
        val_labels: Validation labels.

    Returns:
        results: Dict with accuracy, test results, and latency.
    """
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    evaluator = TFLiteEvaluator(tflite_path, tokenizer, label_map)

    # -------------------------------------------------------
    # Validation accuracy
    # -------------------------------------------------------
    accuracy = evaluator.run_validation(val_texts, val_labels, label_encoder)
    print(f"  Val accuracy (TFLite): {accuracy:.1f}%")

    # -------------------------------------------------------
    # English tests
    # -------------------------------------------------------
    en_tests = [
        ("How do I reset my password?", "account_password"),
        ("Cancel my subscription", "subscription_cancel"),
        ("The app keeps crashing", "bug_report"),
        ("What's the weather?", "out_of_scope"),
        ("I need to talk to someone", "need_human"),
    ]

    en_header = 'English Tests (lang="en")'
    print(f"\n  {en_header:=^50}")
    en_results = []
    for text, expected in en_tests:
        intent, conf = evaluator.predict(text, lang='en')
        passed = intent == expected
        status = 'PASS' if passed else 'FAIL'
        print(f"    [{status}] \"{text}\" → {intent} ({conf:.2f})")
        en_results.append((text, expected, intent, conf, passed))

    # -------------------------------------------------------
    # Japanese tests
    # -------------------------------------------------------
    ja_tests = [
        ("パスワードを変更したい", "account_password"),
        ("解約したい", "subscription_cancel"),
        ("アプリがフリーズする", "bug_report"),
        ("今日の天気は？", "out_of_scope"),
        ("サポートに連絡したい", "need_human"),
    ]

    ja_header = 'Japanese Tests (lang="ja")'
    print(f"\n  {ja_header:=^50}")
    ja_results = []
    for text, expected in ja_tests:
        intent, conf = evaluator.predict(text, lang='ja')
        passed = intent == expected
        status = 'PASS' if passed else 'FAIL'
        print(f"    [{status}] \"{text}\" → {intent} ({conf:.2f})")
        ja_results.append((text, expected, intent, conf, passed))

    # -------------------------------------------------------
    # Latency
    # -------------------------------------------------------
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        evaluator.predict("How do I reset my password?", lang='en')
        times.append((time.perf_counter() - t0) * 1000)

    median_ms = np.median(times)
    print(f"\n  TFLite latency: {median_ms:.1f}ms median (CPU)")

    return {
        'accuracy': accuracy,
        'en_tests': en_results,
        'ja_tests': ja_results,
        'latency_ms': median_ms,
    }
