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

from scripts.config import MAX_SEQ_LENGTH, LANG_PREFIX, CONFIDENCE_THRESHOLD, CLARIFY_GAP


class TFLiteEvaluator:
    """Evaluates a TFLite model with bilingual test cases."""

    def __init__(self, tflite_path, tokenizer, label_map, responses=None, qa_matcher=None):
        """
        Args:
            tflite_path: Path to .tflite model file.
            tokenizer: HuggingFace tokenizer.
            label_map: Dict[int, str] mapping label ID → tag name.
            responses: Optional dict from responses.json (tag → {en, ja, type}).
            qa_matcher: Optional QAMatcher for smart response matching.
        """
        self.tokenizer = tokenizer
        self.label_map = label_map
        self._responses = responses or {}
        self._qa_matcher = qa_matcher

        # Lazy import — tensorflow only loaded when evaluator is used
        import tensorflow as tf

        # Load interpreter
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.inp_details = self.interpreter.get_input_details()
        self.out_details = self.interpreter.get_output_details()

        # Map input names → (index, dtype). onnx2tf may reorder inputs.
        self._inputs = {}
        for detail in self.inp_details:
            name = detail['name'].lower()
            dtype = detail['dtype']
            if 'input_id' in name:
                self._inputs['input_ids'] = (detail['index'], dtype)
            elif 'attention' in name or 'mask' in name:
                self._inputs['attention_mask'] = (detail['index'], dtype)
        # Fallback: assume positional order if names don't match
        if 'input_ids' not in self._inputs:
            self._inputs['input_ids'] = (self.inp_details[0]['index'], self.inp_details[0]['dtype'])
            self._inputs['attention_mask'] = (self.inp_details[1]['index'], self.inp_details[1]['dtype'])

    def _run_inference(self, text, lang='en'):
        """Run raw inference, return probability array."""
        full_text = f"{LANG_PREFIX[lang]} {text}"
        enc = self.tokenizer(
            full_text,
            return_tensors='np',
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

        ids_idx, ids_dtype = self._inputs['input_ids']
        mask_idx, mask_dtype = self._inputs['attention_mask']
        self.interpreter.set_tensor(ids_idx, enc['input_ids'].astype(ids_dtype))
        self.interpreter.set_tensor(mask_idx, enc['attention_mask'].astype(mask_dtype))
        self.interpreter.invoke()

        logits = self.interpreter.get_tensor(self.out_details[0]['index'])
        logits_stable = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits_stable) / np.exp(logits_stable).sum(axis=-1, keepdims=True)
        return probs[0]

    def predict(self, text, lang='en', threshold=None):
        """
        Run inference on a single text.

        Returns:
            (tag, confidence): Predicted tag and confidence score.
        """
        threshold = threshold if threshold is not None else CONFIDENCE_THRESHOLD
        probs = self._run_inference(text, lang)
        pred_id = np.argmax(probs)
        confidence = float(probs[pred_id])
        tag = self.label_map[int(pred_id)]

        if confidence < threshold:
            return 'unknown', confidence
        return tag, confidence

    def predict_top_n(self, text, lang='en', n=3):
        """
        Return top-N predictions sorted by confidence.

        Returns:
            List of (tag, confidence) tuples, highest first.
        """
        probs = self._run_inference(text, lang)
        top_ids = np.argsort(probs)[::-1][:n]
        return [(self.label_map[int(i)], float(probs[i])) for i in top_ids]

    def get_response(self, text, lang='en', threshold=None, clarify_gap=None):
        """
        Classify text and return the best matching response.

        Flow:
        1. If confident → return single response (answer/action_*/support)
        2. If ambiguous (top-2 scores close) → return "clarify" with options
        3. If low confidence → return "reject" (unknown)

        Returns:
            dict: {type, tag, response_text} or
            dict: {type: "clarify", options: [...], response_text: "..."}
        """
        threshold = threshold if threshold is not None else CONFIDENCE_THRESHOLD
        clarify_gap = clarify_gap if clarify_gap is not None else CLARIFY_GAP

        top = self.predict_top_n(text, lang=lang, n=3)
        top1_tag, top1_conf = top[0]
        top2_tag, top2_conf = top[1] if len(top) > 1 else ('', 0.0)

        # Below threshold → reject
        if top1_conf < threshold:
            resp = self._responses.get('unknown', {})
            return {
                'type': 'reject',
                'tag': 'unknown',
                'response_text': resp.get(lang, ''),
            }

        # Ambiguous: top-2 are close AND both above threshold → ask user to clarify
        gap = top1_conf - top2_conf
        if gap < clarify_gap and top2_conf >= threshold and top1_tag != top2_tag:
            options = []
            for tag, conf in top[:2]:
                resp = self._responses.get(tag, {})
                options.append({
                    'tag': tag,
                    'type': resp.get('type', 'answer'),
                    'label': resp.get(f'label_{lang}', '') or tag.replace('_', ' ').title(),
                    'confidence': round(conf, 2),
                })

            clarify_text = {
                'en': "I'd like to help! Could you tell me which of these you need?",
                'ja': 'お手伝いしたいです！以下のどちらをお求めですか？',
            }
            return {
                'type': 'clarify',
                'tag': 'ambiguous',
                'options': options,
                'response_text': clarify_text.get(lang, clarify_text['en']),
            }

        # Confident → return single response
        tag = top1_tag
        resp = self._responses.get(tag, {})
        resp_type = resp.get('type', 'reject')

        if self._qa_matcher:
            response_text = self._qa_matcher.find_best_answer(text, tag, lang)
        else:
            response_text = ''

        if not response_text:
            response_text = resp.get(lang, '')

        return {
            'type': resp_type,
            'tag': tag,
            'response_text': response_text,
        }

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
            # Strip prefix for predict() — handles both
            # "query: [EN] text" (E5) and "[EN] text" (legacy) formats
            lang = 'en' if '[EN]' in text else 'ja'
            clean = text
            for pfx in ('query: [EN] ', 'query: [JA] ', '[EN] ', '[JA] '):
                if clean.startswith(pfx):
                    clean = clean[len(pfx):]
                    break
            pred_tag, _ = self.predict(clean, lang=lang, threshold=0.0)

            if label_encoder.classes_[label] == pred_tag:
                correct += 1

        accuracy = correct / len(val_texts) * 100
        return accuracy


def evaluate_model(tflite_path, tokenizer, label_map, label_encoder,
                   val_texts, val_labels, responses=None):
    """
    Full evaluation: validation accuracy + bilingual tests + latency.

    Args:
        tflite_path: Path to .tflite model.
        tokenizer: HuggingFace tokenizer.
        label_map: Dict[int, str] label mapping.
        label_encoder: Fitted LabelEncoder.
        val_texts: Validation texts.
        val_labels: Validation labels.
        responses: Optional dict from responses.json.

    Returns:
        results: Dict with accuracy, test results, and latency.
    """
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    evaluator = TFLiteEvaluator(tflite_path, tokenizer, label_map, responses)

    # -------------------------------------------------------
    # Validation accuracy
    # -------------------------------------------------------
    accuracy = evaluator.run_validation(val_texts, val_labels, label_encoder)
    print(f"  Val accuracy (TFLite): {accuracy:.1f}%")

    # -------------------------------------------------------
    # English tests
    # -------------------------------------------------------
    en_tests = [
        ("How do I reset my password?", "password"),
        ("Cancel my subscription", "subscription"),
        ("The app keeps crashing", "bug"),
        ("What's the weather?", "unknown"),
        ("I need to talk to someone", "support"),
    ]

    en_header = 'English Tests (lang="en")'
    print(f"\n  {en_header:=^50}")
    en_results = []
    for text, expected in en_tests:
        tag, conf = evaluator.predict(text, lang='en')
        passed = tag == expected
        status = 'PASS' if passed else 'FAIL'
        print(f"    [{status}] \"{text}\" → {tag} ({conf:.2f})")
        en_results.append((text, expected, tag, conf, passed))

    # -------------------------------------------------------
    # Japanese tests
    # -------------------------------------------------------
    ja_tests = [
        ("パスワードを変更したい", "password"),
        ("解約したい", "subscription"),
        ("アプリがフリーズする", "bug"),
        ("今日の天気は？", "unknown"),
        ("サポートに連絡したい", "support"),
    ]

    ja_header = 'Japanese Tests (lang="ja")'
    print(f"\n  {ja_header:=^50}")
    ja_results = []
    for text, expected in ja_tests:
        tag, conf = evaluator.predict(text, lang='ja')
        passed = tag == expected
        status = 'PASS' if passed else 'FAIL'
        print(f"    [{status}] \"{text}\" → {tag} ({conf:.2f})")
        ja_results.append((text, expected, tag, conf, passed))

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
