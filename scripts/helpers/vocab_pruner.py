# ===========================================================
# vocab_pruner.py — Prune tokenizer to EN+JP only
# ===========================================================
"""
Strips unused language tokens from the embedding matrix.
Reduces vocab from ~256K → ~35K (EN + JP only).
Saves token ID mapping for tokenizer reconstruction.
"""

import gc
import json
import os

import numpy as np
import torch
from transformers import AutoModel

from scripts.config import BASE_MODEL, DTYPE, EXPORT_DIR


def prune_vocabulary(tokenizer, texts, export_dir=None):
    """
    Prune the base model vocabulary to keep only EN + JP tokens.

    Strategy:
      1. Collect all token IDs used in training data
      2. Add special tokens + basic ASCII characters
      3. Rebuild a smaller embedding matrix
      4. Save old→new token ID mapping

    Args:
        tokenizer: HuggingFace tokenizer.
        texts: All training text samples (for token collection).
        export_dir: Override export directory.

    Returns:
        base_model: Model with pruned embeddings.
        kept_ids: Sorted list of kept original token IDs.
        old_to_new: Dict mapping old token ID → new token ID.
    """
    export_dir = export_dir or EXPORT_DIR

    print("=" * 60)
    print("VOCAB PRUNING")
    print("=" * 60)

    # Load base model
    base_model = AutoModel.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)
    original_vocab = tokenizer.vocab_size
    original_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Original: {original_vocab:,} vocab, {original_params:,} params")

    # -------------------------------------------------------
    # Collect used tokens
    # -------------------------------------------------------
    used_tokens = set()

    # From all training patterns
    for text in texts:
        used_tokens.update(tokenizer.encode(text, add_special_tokens=False))

    # All special tokens
    for sid in tokenizer.all_special_ids:
        used_tokens.add(sid)

    # Basic ASCII + common symbols
    ascii_chars = (
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        '.,!?;:\'"()-/@#$%& '
    )
    for char in ascii_chars:
        used_tokens.update(tokenizer.encode(char, add_special_tokens=False))

    kept_ids = sorted(used_tokens)
    ratio = len(kept_ids) / original_vocab * 100
    print(f"  Pruned vocab: {len(kept_ids):,} ({ratio:.1f}% of original)")

    # -------------------------------------------------------
    # Rebuild embedding matrix
    # -------------------------------------------------------
    old_emb = base_model.embeddings.word_embeddings.weight.data
    new_emb = torch.nn.Embedding(len(kept_ids), old_emb.shape[1])
    for new_id, old_id in enumerate(kept_ids):
        new_emb.weight.data[new_id] = old_emb[old_id]

    base_model.embeddings.word_embeddings = new_emb
    base_model.config.vocab_size = len(kept_ids)

    # -------------------------------------------------------
    # Save token ID mapping
    # -------------------------------------------------------
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}
    map_path = os.path.join(export_dir, 'token_id_map.json')
    with open(map_path, 'w') as f:
        json.dump({str(k): v for k, v in old_to_new.items()}, f)

    pruned_params = sum(p.numel() for p in base_model.parameters())
    reduction = (1 - pruned_params / original_params) * 100
    print(f"  Pruned: {pruned_params:,} params ({reduction:.0f}% reduction)")
    print(f"  Saved: token_id_map.json")

    del old_emb
    gc.collect()

    return base_model, kept_ids, old_to_new


class RemappedTokenizer:
    """Wraps a HuggingFace tokenizer to remap IDs to a pruned vocabulary.

    After vocab pruning, the embedding matrix uses new contiguous IDs (0..N),
    but the original tokenizer still produces old IDs from the full 256K vocab.
    This wrapper transparently remaps token IDs so that downstream code
    (training, evaluation, export) receives IDs matching the pruned embeddings.
    """

    def __init__(self, tokenizer, old_to_new):
        self._tokenizer = tokenizer
        self._old_to_new = old_to_new
        self._unk_id = old_to_new.get(tokenizer.unk_token_id, 0)

    def __call__(self, *args, **kwargs):
        output = self._tokenizer(*args, **kwargs)
        output['input_ids'] = self._remap(output['input_ids'])
        return output

    def _remap(self, ids):
        """Remap token IDs from original vocab to pruned vocab."""
        if isinstance(ids, np.ndarray):
            flat = ids.flatten().tolist()
            remapped = [self._old_to_new.get(tid, self._unk_id) for tid in flat]
            return np.array(remapped, dtype=ids.dtype).reshape(ids.shape)
        if isinstance(ids, list):
            if len(ids) > 0 and isinstance(ids[0], list):
                return [
                    [self._old_to_new.get(tid, self._unk_id) for tid in row]
                    for row in ids
                ]
            return [self._old_to_new.get(tid, self._unk_id) for tid in ids]
        return ids

    def encode(self, *args, **kwargs):
        ids = self._tokenizer.encode(*args, **kwargs)
        return [self._old_to_new.get(tid, self._unk_id) for tid in ids]

    def save_pretrained(self, *args, **kwargs):
        return self._tokenizer.save_pretrained(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)
