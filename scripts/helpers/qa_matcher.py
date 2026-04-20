# ===========================================================
# qa_matcher.py — Token-overlap matching for smart responses
# ===========================================================
"""
Finds the best matching Q&A pair within a classified tag
using token-level Jaccard similarity. Runs on-device with
no extra model — just tokenizer + pre-indexed token sets.
"""

import json
import os
from collections import defaultdict


class QAMatcher:
    """Matches user queries to the closest training Q&A pair."""

    def __init__(self, qa_index_path, tokenizer):
        with open(qa_index_path, 'r', encoding='utf-8') as f:
            self._index = json.load(f)
        self._tokenizer = tokenizer
        self._token_cache = {}
        self._build_token_cache()

    def _tokenize(self, text):
        return set(self._tokenizer.encode(text, add_special_tokens=False))

    def _build_token_cache(self):
        for tag, pairs in self._index.items():
            for i, pair in enumerate(pairs):
                for lang in ('en', 'ja'):
                    q = pair.get(f'q_{lang}', '')
                    if q:
                        self._token_cache[(tag, i, lang)] = self._tokenize(q)

    def find_best_answer(self, query, tag, lang='en'):
        """
        Find the best matching answer for a query within a tag.

        Args:
            query: Raw user query text.
            tag: Classified tag name.
            lang: 'en' or 'ja'.

        Returns:
            str: Best matching answer text, or empty string if no match.
        """
        pairs = self._index.get(tag, [])
        if not pairs:
            return ''

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return pairs[0].get(f'a_{lang}', '')

        best_score = -1
        best_answer = ''

        for i, pair in enumerate(pairs):
            cached = self._token_cache.get((tag, i, lang))
            if cached is None:
                continue

            # Jaccard similarity
            intersection = len(query_tokens & cached)
            union = len(query_tokens | cached)
            score = intersection / union if union > 0 else 0

            if score > best_score:
                best_score = score
                best_answer = pair.get(f'a_{lang}', '')

        return best_answer if best_answer else pairs[0].get(f'a_{lang}', '')
