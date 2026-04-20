"""
Demo web server for the bilingual Support AI model.

Usage:
    python demo/app.py
    → Open http://localhost:5050
"""

import json
import os
import sys

from flask import Flask, jsonify, request, send_from_directory

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.config import EXPORT_DIR, BASE_MODEL
from scripts.helpers.evaluator import TFLiteEvaluator
from scripts.helpers.vocab_pruner import RemappedTokenizer
from transformers import AutoTokenizer

app = Flask(__name__, static_folder='static')

# ── Load model once at startup ──────────────────────────────────
print("Loading model...")

with open(os.path.join(EXPORT_DIR, 'label_map.json')) as f:
    label_map = {int(k): v for k, v in json.load(f).items()}

with open(os.path.join(EXPORT_DIR, 'responses.json'), encoding='utf-8') as f:
    responses = json.load(f)

with open(os.path.join(EXPORT_DIR, 'token_id_map.json')) as f:
    old_to_new = {int(k): int(v) for k, v in json.load(f).items()}

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
remapped = RemappedTokenizer(tokenizer, old_to_new)

tflite_path = os.path.join(EXPORT_DIR, 'support_ai.tflite')
evaluator = TFLiteEvaluator(tflite_path, remapped, label_map, responses)

print(f"  Model loaded: {len(label_map)} tags")
print(f"  Tags: {list(label_map.values())}")
print("  Ready → http://localhost:5050")


# ── Routes ──────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()
    lang = data.get('lang', 'en')

    if not text:
        return jsonify({'error': 'empty text'}), 400

    tag, confidence = evaluator.predict(text, lang=lang, threshold=0.0)

    resp = responses.get(tag, {})
    response_text = resp.get(lang, '')
    resp_type = resp.get('type', 'reject')

    # Apply confidence threshold — low confidence → unknown
    # E5-small typically outputs 0.40-0.88 for correct predictions
    threshold = 0.35
    effective_tag = tag if confidence >= threshold else 'unknown'
    if effective_tag == 'unknown' and tag != 'unknown':
        resp = responses.get('unknown', {})
        response_text = resp.get(lang, '')
        resp_type = 'reject'

    return jsonify({
        'tag': str(effective_tag),
        'raw_tag': str(tag),
        'confidence': round(float(confidence), 4),
        'threshold': threshold,
        'above_threshold': bool(confidence >= threshold),
        'type': str(resp_type),
        'response': str(response_text),
    })


@app.route('/api/tags', methods=['GET'])
def tags():
    return jsonify({
        'tags': list(label_map.values()),
        'responses': responses,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
