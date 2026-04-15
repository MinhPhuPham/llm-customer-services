# ===========================================================
# export_artifacts.py — Save label_map.json, responses.json
# ===========================================================
"""
Exports artifacts needed by the mobile app:
  - label_map.json: intent_id → intent_name mapping
  - responses.json: intent → {en, ja, type} response templates
"""

import json
import os

import pandas as pd

from scripts.config import EXPORT_DIR


def export_label_map(label_encoder, export_dir=None):
    """
    Save label_map.json (intent_id → intent_name).

    Args:
        label_encoder: Fitted sklearn LabelEncoder.
        export_dir: Override export directory (default: config.EXPORT_DIR).

    Returns:
        label_map: Dict mapping int → intent name.
    """
    export_dir = export_dir or EXPORT_DIR

    label_map = {
        int(i): label
        for i, label in enumerate(label_encoder.classes_)
    }

    path = os.path.join(export_dir, 'label_map.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"  Saved: label_map.json ({len(label_map)} intents)")
    return label_map


def export_responses(df, export_dir=None):
    """
    Save responses.json (intent → {en, ja, type}).

    Args:
        df: DataFrame with columns [intent, type, q_en, q_ja, a_en, a_ja].
        export_dir: Override export directory (default: config.EXPORT_DIR).

    Returns:
        responses: Dict of response templates.
    """
    export_dir = export_dir or EXPORT_DIR

    responses = {}
    for _, row in df.drop_duplicates('intent').iterrows():
        responses[row['intent']] = {
            'en': str(row['a_en']) if pd.notna(row['a_en']) else '',
            'ja': str(row['a_ja']) if pd.notna(row['a_ja']) else '',
            'type': row['type'],
        }

    path = os.path.join(export_dir, 'responses.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    print(f"  Saved: responses.json ({len(responses)} intents)")
    return responses
