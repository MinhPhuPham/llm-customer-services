#!/usr/bin/env python3
"""
Interactive chat — test the model like a real user.

Usage:
    python test/chat.py                    # English mode (default)
    python test/chat.py --lang ja          # Japanese mode
    python test/chat.py --no-color         # Plain output (no ANSI colors)

Commands inside chat:
    /en          Switch to English
    /ja          Switch to Japanese
    /lang        Show current language
    /tags        Show all known tags
    /threshold N Set confidence threshold (e.g. /threshold 0.5)
    /debug       Toggle debug mode (show confidence + all scores)
    /q or /quit  Exit
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def load_model():
    from scripts.config import EXPORT_DIR

    tflite_path = os.path.join(EXPORT_DIR, 'support_ai.tflite')
    label_map_path = os.path.join(EXPORT_DIR, 'label_map.json')
    responses_path = os.path.join(EXPORT_DIR, 'responses.json')
    token_map_path = os.path.join(EXPORT_DIR, 'token_id_map.json')

    missing = [p for p in [tflite_path, label_map_path, responses_path, token_map_path]
               if not os.path.exists(p)]
    if missing:
        print(f"Missing files: {missing}")
        print("Train and export the model first (run notebook on Colab).")
        sys.exit(1)

    with open(label_map_path) as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    with open(responses_path) as f:
        responses = json.load(f)
    with open(token_map_path) as f:
        old_to_new = {int(k): v for k, v in json.load(f).items()}

    from transformers import AutoTokenizer
    from scripts.config import BASE_MODEL
    from scripts.helpers.vocab_pruner import RemappedTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer = RemappedTokenizer(tokenizer, old_to_new)

    from scripts.helpers.evaluator import TFLiteEvaluator
    evaluator = TFLiteEvaluator(tflite_path, tokenizer, label_map, responses)

    return evaluator, label_map, responses


class Colors:
    BOLD = '\033[1m'
    DIM = '\033[2m'
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[0;33m'
    CYAN = '\033[0;36m'
    RED = '\033[0;31m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'


class NoColors:
    BOLD = DIM = GREEN = BLUE = YELLOW = CYAN = RED = MAGENTA = NC = ''


TYPE_COLORS = {
    'answer': 'GREEN',
    'support': 'YELLOW',
    'reject': 'RED',
}


def main():
    parser = argparse.ArgumentParser(description='Interactive model chat')
    parser.add_argument('--lang', default='en', choices=['en', 'ja'])
    parser.add_argument('--no-color', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()

    C = NoColors if args.no_color else Colors
    lang = args.lang
    debug = args.debug
    threshold = args.threshold

    print(f"\n{C.BOLD}Loading model...{C.NC}", end=' ', flush=True)
    evaluator, label_map, responses = load_model()
    print(f"{C.GREEN}OK{C.NC}")

    tags = sorted(set(label_map.values()))
    lang_name = {'en': 'English', 'ja': 'Japanese'}

    print(f"""
{C.BOLD}{'=' * 50}
  Support AI — Interactive Chat
{'=' * 50}{C.NC}
  Language:  {C.CYAN}{lang_name[lang]}{C.NC}  {C.DIM}(/en /ja to switch){C.NC}
  Tags:      {', '.join(tags)}
  Threshold: {threshold if threshold is not None else 'default (0.85)'}
  Debug:     {'on' if debug else f'off  {C.DIM}(/debug to toggle){C.NC}'}

  {C.DIM}Type a question as if you're a user.
  Commands: /en /ja /lang /tags /threshold N /debug /q{C.NC}
""")

    while True:
        try:
            prompt_char = 'あなた' if lang == 'ja' else 'You'
            user_input = input(f"  {C.BOLD}{prompt_char}:{C.NC} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C.DIM}Bye!{C.NC}\n")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith('/'):
            cmd = user_input.lower().split()
            if cmd[0] in ('/q', '/quit', '/exit'):
                print(f"  {C.DIM}Bye!{C.NC}\n")
                break
            elif cmd[0] == '/en':
                lang = 'en'
                print(f"  {C.CYAN}Switched to English{C.NC}\n")
                continue
            elif cmd[0] == '/ja':
                lang = 'ja'
                print(f"  {C.CYAN}Switched to Japanese{C.NC}\n")
                continue
            elif cmd[0] == '/lang':
                print(f"  {C.CYAN}Current: {lang_name[lang]}{C.NC}\n")
                continue
            elif cmd[0] == '/tags':
                for t in tags:
                    r = responses.get(t, {})
                    print(f"  {C.BOLD}{t:15s}{C.NC} [{r.get('type', '?')}]")
                print()
                continue
            elif cmd[0] == '/threshold':
                if len(cmd) > 1:
                    try:
                        threshold = float(cmd[1])
                        print(f"  {C.CYAN}Threshold set to {threshold}{C.NC}\n")
                    except ValueError:
                        print(f"  {C.RED}Invalid number{C.NC}\n")
                else:
                    cur = threshold if threshold is not None else 'default (0.85)'
                    print(f"  {C.CYAN}Current threshold: {cur}{C.NC}\n")
                continue
            elif cmd[0] == '/debug':
                debug = not debug
                print(f"  {C.CYAN}Debug: {'on' if debug else 'off'}{C.NC}\n")
                continue
            else:
                print(f"  {C.DIM}Unknown command. Try: /en /ja /tags /threshold /debug /q{C.NC}\n")
                continue

        # Inference
        t0 = time.perf_counter()
        tag, confidence = evaluator.predict(user_input, lang=lang, threshold=threshold)
        latency = (time.perf_counter() - t0) * 1000

        resp = responses.get(tag, {})
        resp_type = resp.get('type', 'reject')
        resp_text = resp.get(lang, '')
        type_color = getattr(C, TYPE_COLORS.get(resp_type, 'NC'))

        # Response
        print()
        print(f"  {C.BOLD}  Bot:{C.NC} {resp_text}")
        print(f"  {C.DIM}       [{type_color}{resp_type}{C.NC}{C.DIM}] tag={tag}  conf={confidence:.2f}  {latency:.0f}ms{C.NC}")

        if debug:
            # Show all tag probabilities
            full_text = f"{evaluator.tokenizer._tokenizer.pad_token or ''}"
            import numpy as np
            from scripts.config import LANG_PREFIX, MAX_SEQ_LENGTH

            full_text = f"{LANG_PREFIX[lang]} {user_input}"
            enc = evaluator.tokenizer(
                full_text, return_tensors='np',
                padding='max_length', truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )
            ids_idx, ids_dtype = evaluator._inputs['input_ids']
            mask_idx, mask_dtype = evaluator._inputs['attention_mask']
            evaluator.interpreter.set_tensor(ids_idx, enc['input_ids'].astype(ids_dtype))
            evaluator.interpreter.set_tensor(mask_idx, enc['attention_mask'].astype(mask_dtype))
            evaluator.interpreter.invoke()
            logits = evaluator.interpreter.get_tensor(evaluator.out_details[0]['index'])
            logits_stable = logits - logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits_stable) / np.exp(logits_stable).sum(axis=-1, keepdims=True)

            scored = [(label_map[i], float(probs[0][i])) for i in range(len(label_map))]
            scored.sort(key=lambda x: -x[1])
            print(f"  {C.DIM}       scores: ", end='')
            parts = [f"{t}={p:.2f}" for t, p in scored]
            print(f"{' | '.join(parts)}{C.NC}")

        print()


if __name__ == '__main__':
    main()
