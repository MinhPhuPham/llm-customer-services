#!/usr/bin/env python3
"""
Test suite for the bilingual customer support AI pipeline.

Tests are organized in layers:
  1. Data pipeline       — Excel parsing, label encoding, responses (no GPU needed)
  2. Tokenization        — Vocab pruning, remapped tokenizer (needs model download)
  3. Inference           — TFLite model predictions (needs trained model)
  4. Conversation flow   — Multi-turn scenarios: greeting → question → answer

Usage:
    python -m pytest test/test_pipeline.py -v                    # all tests
    python -m pytest test/test_pipeline.py -v -k "data"          # data tests only
    python -m pytest test/test_pipeline.py -v -k "tokenize"      # tokenizer tests
    python -m pytest test/test_pipeline.py -v -k "inference"     # inference tests
    python -m pytest test/test_pipeline.py -v -k "conversation"  # conversation flow
    python -m pytest test/test_pipeline.py -v -k "greeting"      # greeting-related
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXCEL_PATH = os.path.join(PROJECT_ROOT, 'service_model_training_data_template.xlsx')
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CASES_PATH = os.path.join(TEST_DIR, 'test_cases.json')

with open(TEST_CASES_PATH, 'r', encoding='utf-8') as f:
    TEST_CASES = json.load(f)


def has_excel():
    return os.path.exists(EXCEL_PATH)


def has_tflite():
    try:
        from scripts.config import EXPORT_DIR
        return os.path.exists(os.path.join(EXPORT_DIR, 'support_ai.tflite'))
    except Exception:
        return False


# =================================================================
# Fixtures
# =================================================================

@pytest.fixture(scope='session')
def parsed_data():
    from scripts.data_loader.excel_parser import parse_excel
    train_rows, df = parse_excel(EXCEL_PATH)
    return train_rows, df


@pytest.fixture(scope='session')
def splits(parsed_data):
    from scripts.data_loader.dataset_builder import prepare_splits
    train_rows, _ = parsed_data
    return prepare_splits(train_rows)


@pytest.fixture(scope='session')
def export_artifacts(parsed_data, splits):
    _, df = parsed_data
    _, _, _, _, _, label_encoder, _ = splits

    from scripts.data_loader.export_artifacts import export_label_map, export_responses
    with tempfile.TemporaryDirectory() as tmp:
        label_map = export_label_map(label_encoder, export_dir=tmp)
        responses = export_responses(df, export_dir=tmp)

        with open(os.path.join(tmp, 'label_map.json')) as f:
            label_map_json = json.load(f)
        with open(os.path.join(tmp, 'responses.json')) as f:
            responses_json = json.load(f)

        yield label_map, responses, label_map_json, responses_json


@pytest.fixture(scope='session')
def tokenizer_and_model(splits):
    texts = splits[0]
    from transformers import AutoTokenizer
    from scripts.config import BASE_MODEL
    from scripts.helpers.vocab_pruner import prune_vocabulary, RemappedTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    with tempfile.TemporaryDirectory() as tmp:
        base_model, kept_ids, old_to_new = prune_vocabulary(
            tokenizer, texts, export_dir=tmp,
        )
    remapped = RemappedTokenizer(tokenizer, old_to_new)
    return remapped, base_model, kept_ids, old_to_new, tokenizer


@pytest.fixture(scope='session')
def tflite_evaluator(tokenizer_and_model, export_artifacts):
    from scripts.config import EXPORT_DIR
    tflite_path = os.path.join(EXPORT_DIR, 'support_ai.tflite')
    if not os.path.exists(tflite_path):
        pytest.skip('TFLite model not found — train and export first')

    remapped, _, _, _, _ = tokenizer_and_model
    label_map, _, _, _ = export_artifacts
    from scripts.helpers.evaluator import TFLiteEvaluator
    return TFLiteEvaluator(tflite_path, remapped, label_map)


# =================================================================
# 1. DATA PIPELINE TESTS
# =================================================================

@pytest.mark.skipif(not has_excel(), reason='Excel file not found')
class TestDataPipeline:

    def test_dp01_parse_produces_train_rows(self, parsed_data):
        train_rows, _ = parsed_data
        assert len(train_rows) >= 2
        for text, intent in train_rows:
            assert isinstance(text, str) and len(text) > 0
            assert isinstance(intent, str) and len(intent) > 0

    def test_dp02_language_prefixes(self, parsed_data):
        train_rows, _ = parsed_data
        for text, _ in train_rows:
            assert '[EN]' in text or '[JA]' in text, \
                f'Missing language tag: "{text[:50]}"'

    def test_dp03_bilingual_coverage(self, parsed_data):
        train_rows, _ = parsed_data
        en = sum(1 for t, _ in train_rows if '[EN]' in t)
        ja = sum(1 for t, _ in train_rows if '[JA]' in t)
        assert en > 0, 'No English samples'
        assert ja > 0, 'No Japanese samples'

    def test_dp04_label_encoder(self, splits):
        _, _, _, _, _, le, num = splits
        assert num >= 2
        assert len(le.classes_) == num

    def test_dp05_train_val_split(self, splits):
        _, train_t, val_t, train_l, val_l, _, _ = splits
        assert len(train_t) > 0 and len(val_t) > 0
        tc = set(train_l.tolist() if hasattr(train_l, 'tolist') else train_l)
        vc = set(val_l.tolist() if hasattr(val_l, 'tolist') else val_l)
        assert tc == vc, f'Train classes {tc} != val classes {vc}'

    def test_dp06_label_map_json(self, export_artifacts):
        _, _, lm, _ = export_artifacts
        assert len(lm) >= 2
        for k, v in lm.items():
            assert k.isdigit()
            assert isinstance(v, str) and len(v) > 0

    def test_dp07_responses_json(self, export_artifacts):
        _, _, _, rj = export_artifacts
        valid_types = {'answer', 'support', 'reject'}
        for intent, resp in rj.items():
            assert 'en' in resp, f'{intent} missing "en"'
            assert 'ja' in resp, f'{intent} missing "ja"'
            assert 'type' in resp, f'{intent} missing "type"'
            assert resp['type'] in valid_types

    def test_dp08_all_intents_have_responses(self, export_artifacts):
        _, _, lm, rj = export_artifacts
        for iid, name in lm.items():
            assert name in rj, f'Intent "{name}" (id={iid}) not in responses.json'

    def test_dp09_answer_types_have_content(self, export_artifacts):
        _, _, _, rj = export_artifacts
        for intent, resp in rj.items():
            if resp['type'] == 'answer':
                assert resp['en'].strip(), f'{intent}: EN answer empty'
                assert resp['ja'].strip(), f'{intent}: JA answer empty'

    def test_dp10_greeting_intent_exists(self, parsed_data):
        """Training data must include a greeting intent for conversation flow."""
        train_rows, _ = parsed_data
        intents = {intent for _, intent in train_rows}
        assert 'greeting' in intents, (
            'Missing "greeting" intent in training data. '
            'Add rows like: intent=greeting, q_en="hi", q_en="hello", '
            'a_en="Hello! I\'m the support bot..."'
        )

    def test_dp11_greeting_has_response(self, export_artifacts):
        """greeting intent must have a welcome message in responses.json."""
        _, _, _, rj = export_artifacts
        assert 'greeting' in rj, 'greeting not in responses.json'
        g = rj['greeting']
        assert g['type'] == 'answer', f'greeting type should be "answer", got "{g["type"]}"'
        assert g['en'].strip(), 'greeting EN response is empty'
        assert g['ja'].strip(), 'greeting JA response is empty'


# =================================================================
# 2. TOKENIZATION TESTS
# =================================================================

@pytest.mark.skipif(not has_excel(), reason='Excel file not found')
class TestTokenization:

    def test_tk01_vocab_pruning_reduces_size(self, tokenizer_and_model):
        _, _, kept_ids, _, orig = tokenizer_and_model
        ratio = len(kept_ids) / orig.vocab_size
        assert ratio < 0.20, f'Pruned ratio {ratio:.2%} >= 20%'

    def test_tk02_remapped_ids_in_range(self, tokenizer_and_model):
        remapped, _, kept_ids, _, _ = tokenizer_and_model
        texts = ['[EN] How do I reset my password?', '[JA] パスワードを変更したい']
        vs = len(kept_ids)
        for text in texts:
            ids = remapped.encode(text, add_special_tokens=True)
            for tid in ids:
                assert 0 <= tid < vs, f'ID {tid} out of [0, {vs})'

    def test_tk03_tokenized_shape(self, tokenizer_and_model):
        from scripts.config import MAX_SEQ_LENGTH
        remapped, _, _, _, _ = tokenizer_and_model
        out = remapped(
            '[EN] test input', return_tensors='np',
            padding='max_length', truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        assert out['input_ids'].shape == (1, MAX_SEQ_LENGTH)
        assert out['attention_mask'].shape == (1, MAX_SEQ_LENGTH)

    def test_tk04_special_tokens_preserved(self, tokenizer_and_model):
        _, _, _, o2n, orig = tokenizer_and_model
        for sid in orig.all_special_ids:
            assert sid in o2n, f'Special token {sid} missing from pruned vocab'

    def test_tk05_tokenize_datasets(self, splits, tokenizer_and_model):
        from scripts.data_loader.dataset_builder import tokenize_datasets
        _, tt, vt, tl, vl, _, _ = splits
        remapped, _, _, _, _ = tokenizer_and_model
        train_ds, val_ds = tokenize_datasets(tt, vt, tl, vl, remapped)
        assert len(train_ds) == len(tt)
        assert len(val_ds) == len(vt)
        for col in ('input_ids', 'attention_mask', 'label'):
            assert col in train_ds.column_names


# =================================================================
# 3. INFERENCE TESTS (require exported TFLite model)
# =================================================================

@pytest.mark.skipif(not has_excel(), reason='Excel file not found')
class TestInference:

    @pytest.mark.parametrize(
        'case',
        TEST_CASES['inference']['english'],
        ids=[c['id'] for c in TEST_CASES['inference']['english']],
    )
    def test_english_intent(self, tflite_evaluator, case):
        intent, conf = tflite_evaluator.predict(
            case['text'], lang=case['lang'], threshold=0.0,
        )
        if case['expected_intent'] == 'out_of_scope':
            assert conf < 0.85, \
                f'{case["id"]}: out_of_scope should have low confidence, got {conf:.2f}'
        else:
            assert intent == case['expected_intent'], \
                f'{case["id"]}: expected {case["expected_intent"]}, got {intent} ({conf:.2f})'
            assert conf >= case['min_confidence'], \
                f'{case["id"]}: confidence {conf:.2f} < {case["min_confidence"]}'

    @pytest.mark.parametrize(
        'case',
        TEST_CASES['inference']['japanese'],
        ids=[c['id'] for c in TEST_CASES['inference']['japanese']],
    )
    def test_japanese_intent(self, tflite_evaluator, case):
        intent, conf = tflite_evaluator.predict(
            case['text'], lang=case['lang'], threshold=0.0,
        )
        if case['expected_intent'] == 'out_of_scope':
            assert conf < 0.85, \
                f'{case["id"]}: out_of_scope should have low confidence, got {conf:.2f}'
        else:
            assert intent == case['expected_intent'], \
                f'{case["id"]}: expected {case["expected_intent"]}, got {intent} ({conf:.2f})'
            assert conf >= case['min_confidence'], \
                f'{case["id"]}: confidence {conf:.2f} < {case["min_confidence"]}'

    def test_confidence_threshold_rejects(self, tflite_evaluator):
        intent, _ = tflite_evaluator.predict(
            "What's the weather?", lang='en', threshold=0.85,
        )
        assert intent == 'out_of_scope'

    def test_confidence_threshold_accepts(self, tflite_evaluator):
        intent, conf = tflite_evaluator.predict(
            'How do I reset my password?', lang='en', threshold=0.85,
        )
        assert intent != 'out_of_scope', f'Expected real intent, got out_of_scope ({conf:.2f})'

    def test_latency_reasonable(self, tflite_evaluator):
        import time
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            tflite_evaluator.predict('test query', lang='en')
            times.append((time.perf_counter() - t0) * 1000)
        median_ms = np.median(times)
        assert median_ms < 100, f'Median latency {median_ms:.1f}ms > 100ms'


# =================================================================
# 4. CONVERSATION FLOW TESTS (require exported TFLite model)
#
# The model is STATELESS — each message is classified independently.
# Multi-turn conversation is managed by the mobile app, not the model.
# These tests verify that each turn produces the correct intent
# so the app can look up the right response from responses.json.
#
# Example flow:
#   User: "hi, can you help me?"
#     → model predicts: greeting
#     → app shows: responses.json["greeting"]["en"]
#       "Hello! I'm the automatic support bot for PROJECT_NAME..."
#
#   User: "i need to change password"
#     → model predicts: account_password
#     → app shows: responses.json["account_password"]["en"]
#       "Yes, I can help you! To change your password: 1. Go to..."
# =================================================================

@pytest.mark.skipif(not has_excel(), reason='Excel file not found')
class TestConversationFlow:

    @pytest.mark.parametrize(
        'scenario',
        TEST_CASES['conversation_flow']['scenarios'],
        ids=[s['id'] for s in TEST_CASES['conversation_flow']['scenarios']],
    )
    def test_conversation_scenario(self, tflite_evaluator, export_artifacts, scenario):
        """Each turn in the conversation should classify to the correct intent."""
        _, responses, _, _ = export_artifacts

        for i, turn in enumerate(scenario['turns']):
            intent, conf = tflite_evaluator.predict(
                turn['user'], lang=turn['lang'], threshold=0.0,
            )

            step = f'{scenario["id"]} turn {i+1}'

            if turn['expected_intent'] == 'out_of_scope':
                assert conf < 0.85, \
                    f'{step}: expected out_of_scope (low conf), got {intent} ({conf:.2f})'
            else:
                assert intent == turn['expected_intent'], \
                    f'{step}: "{turn["user"]}" → expected {turn["expected_intent"]}, got {intent} ({conf:.2f})'

            # Verify the response exists and matches expected type
            if turn['expected_intent'] in responses:
                resp = responses[turn['expected_intent']]
                if 'expected_response_type' in turn:
                    assert resp['type'] == turn['expected_response_type'], \
                        f'{step}: response type should be {turn["expected_response_type"]}, got {resp["type"]}'
                # Answer-type responses must have content in the query language
                if resp['type'] == 'answer':
                    lang_key = turn['lang']
                    assert resp[lang_key].strip(), \
                        f'{step}: {lang_key} response empty for {turn["expected_intent"]}'

    @pytest.mark.parametrize(
        'greeting',
        TEST_CASES['conversation_flow']['greeting_variants']['english'],
        ids=[f'greeting_en_{i}' for i in range(
            len(TEST_CASES['conversation_flow']['greeting_variants']['english'])
        )],
    )
    def test_english_greeting_variants(self, tflite_evaluator, greeting):
        """All common EN greetings should classify as greeting intent."""
        intent, conf = tflite_evaluator.predict(greeting, lang='en', threshold=0.0)
        assert intent == 'greeting', \
            f'"{greeting}" → expected greeting, got {intent} ({conf:.2f})'

    @pytest.mark.parametrize(
        'greeting',
        TEST_CASES['conversation_flow']['greeting_variants']['japanese'],
        ids=[f'greeting_ja_{i}' for i in range(
            len(TEST_CASES['conversation_flow']['greeting_variants']['japanese'])
        )],
    )
    def test_japanese_greeting_variants(self, tflite_evaluator, greeting):
        """All common JA greetings should classify as greeting intent."""
        intent, conf = tflite_evaluator.predict(greeting, lang='ja', threshold=0.0)
        assert intent == 'greeting', \
            f'"{greeting}" → expected greeting, got {intent} ({conf:.2f})'

    def test_greeting_response_is_welcoming(self, export_artifacts):
        """Greeting response should contain a welcome/help message."""
        _, responses, _, _ = export_artifacts
        if 'greeting' not in responses:
            pytest.skip('greeting intent not in responses')
        en = responses['greeting']['en'].lower()
        ja = responses['greeting']['ja']
        assert any(w in en for w in ('hello', 'hi', 'welcome', 'help', 'support', 'bot')), \
            f'Greeting EN response should be welcoming: "{en[:100]}"'
        assert len(ja) > 0, 'Greeting JA response is empty'

    def test_full_conversation_with_response_lookup(self, tflite_evaluator, export_artifacts):
        """
        End-to-end: simulate the full app flow.

        Turn 1: "hi, can you help me?"
          → greeting intent → show welcome message
        Turn 2: "i need to change password"
          → account_password intent → show detailed answer

        This is exactly what the mobile app does.
        """
        _, responses, _, _ = export_artifacts

        # Turn 1: Greeting
        intent1, conf1 = tflite_evaluator.predict(
            'hi, can you help me?', lang='en', threshold=0.85,
        )
        if intent1 != 'out_of_scope' and intent1 in responses:
            resp1 = responses[intent1]
            assert resp1['en'].strip(), f'Empty EN response for {intent1}'

        # Turn 2: Real question
        intent2, conf2 = tflite_evaluator.predict(
            'i need to change password', lang='en', threshold=0.85,
        )
        if intent2 != 'out_of_scope' and intent2 in responses:
            resp2 = responses[intent2]
            assert resp2['type'] == 'answer', \
                f'password question should get an answer, got type={resp2["type"]}'
            assert resp2['en'].strip(), f'Empty EN response for {intent2}'
