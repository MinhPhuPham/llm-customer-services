# ===========================================================
# export_artifacts.py — Save label_map.json, responses.json, qa_index.json
# ===========================================================
"""
Exports artifacts needed by the mobile app:
  - label_map.json: tag_id → tag_name mapping
  - responses.json: tag → {en, ja, type} default response templates
  - qa_index.json: all Q&A pairs per tag for smart response matching
"""

import json
import os

import pandas as pd

from scripts.config import EXPORT_DIR


def export_label_map(label_encoder, export_dir=None):
    """
    Save label_map.json (tag_id → tag_name).

    Args:
        label_encoder: Fitted sklearn LabelEncoder.
        export_dir: Override export directory (default: config.EXPORT_DIR).

    Returns:
        label_map: Dict mapping int → tag name.
    """
    export_dir = export_dir or EXPORT_DIR

    label_map = {
        int(i): label
        for i, label in enumerate(label_encoder.classes_)
    }

    path = os.path.join(export_dir, 'label_map.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"  Saved: label_map.json ({len(label_map)} tags)")
    return label_map


def export_responses(df, export_dir=None):
    """
    Save responses.json (tag → {en, ja, type, label_en, label_ja}).

    Uses handcrafted default templates from default_responses.py.
    Falls back to first Excel row only for tags not defined there.
    The 'type' field from Excel overrides the default if present.

    Args:
        df: DataFrame with columns [tag, type, q_en, q_ja, a_en, a_ja].
        export_dir: Override export directory (default: config.EXPORT_DIR).

    Returns:
        responses: Dict of response templates.
    """
    from scripts.data_loader.default_responses import DEFAULT_RESPONSES

    export_dir = export_dir or EXPORT_DIR

    responses = {}
    for _, row in df.drop_duplicates('tag').iterrows():
        tag = row['tag']
        resp_type = row['type']

        if tag in DEFAULT_RESPONSES:
            entry = dict(DEFAULT_RESPONSES[tag])
            entry['type'] = resp_type
        else:
            entry = {
                'en': str(row['a_en']) if pd.notna(row['a_en']) else '',
                'ja': str(row['a_ja']) if pd.notna(row['a_ja']) else '',
                'type': resp_type,
                'label_en': tag.replace('_', ' ').title(),
                'label_ja': tag.replace('_', ' ').title(),
            }

        if 'label_en' in df.columns and pd.notna(row.get('label_en')):
            entry['label_en'] = str(row['label_en']).strip()
        if 'label_ja' in df.columns and pd.notna(row.get('label_ja')):
            entry['label_ja'] = str(row['label_ja']).strip()

        responses[tag] = entry

    path = os.path.join(export_dir, 'responses.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)

    print(f"  Saved: responses.json ({len(responses)} tags)")
    return responses


def export_qa_index(df, export_dir=None):
    """
    Save qa_index.json — all Q&A pairs grouped by tag for smart matching.

    Each question gets its own specific answer so the bot can return
    the most relevant response instead of a single static template.

    Args:
        df: DataFrame with columns [tag, type, q_en, q_ja, a_en, a_ja].
        export_dir: Override export directory (default: config.EXPORT_DIR).

    Returns:
        qa_index: Dict of tag → list of Q&A pairs.
    """
    export_dir = export_dir or EXPORT_DIR

    qa_index = {}
    for tag in df['tag'].dropna().unique():
        group = df[df['tag'] == tag]
        pairs = []
        for _, row in group.iterrows():
            q_en = str(row['q_en']).strip() if pd.notna(row['q_en']) else ''
            q_ja = str(row['q_ja']).strip() if pd.notna(row['q_ja']) else ''
            a_en = str(row['a_en']).strip() if pd.notna(row['a_en']) else ''
            a_ja = str(row['a_ja']).strip() if pd.notna(row['a_ja']) else ''

            if not q_en and not q_ja:
                continue

            pairs.append({
                'q_en': q_en,
                'q_ja': q_ja,
                'a_en': a_en,
                'a_ja': a_ja,
            })
        qa_index[tag] = pairs

    total_pairs = sum(len(v) for v in qa_index.values())
    path = os.path.join(export_dir, 'qa_index.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(qa_index, f, ensure_ascii=False, indent=2)

    print(f"  Saved: qa_index.json ({total_pairs} Q&A pairs across {len(qa_index)} tags)")
    return qa_index
