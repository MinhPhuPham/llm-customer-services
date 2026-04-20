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
    qa_index_path = os.path.join(EXPORT_DIR, 'qa_index.json')

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

    base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    remapped = RemappedTokenizer(base_tokenizer, old_to_new)

    # Load QA matcher for smart responses (optional)
    qa_matcher = None
    if os.path.exists(qa_index_path):
        from scripts.helpers.qa_matcher import QAMatcher
        qa_matcher = QAMatcher(qa_index_path, base_tokenizer)

    from scripts.helpers.evaluator import TFLiteEvaluator
    evaluator = TFLiteEvaluator(
        tflite_path, remapped, label_map, responses, qa_matcher=qa_matcher,
    )

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
    'clarify': 'MAGENTA',
}


def get_type_color(resp_type):
    if resp_type.startswith('action_'):
        return 'CYAN'
    return TYPE_COLORS.get(resp_type, 'NC')


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

        # Inference — smart response matching
        t0 = time.perf_counter()
        result = evaluator.get_response(user_input, lang=lang, threshold=threshold)
        tag = result['tag']
        resp_type = result['type']
        resp_text = result['response_text']
        _, confidence = evaluator.predict(user_input, lang=lang, threshold=threshold)
        latency = (time.perf_counter() - t0) * 1000
        type_color = getattr(C, get_type_color(resp_type))

        # Response
        print()
        if resp_type == 'clarify':
            print(f"  {C.BOLD}  Bot:{C.NC} {resp_text}")
            for i, opt in enumerate(result.get('options', []), 1):
                opt_color = getattr(C, get_type_color(opt['type']))
                print(f"  {C.BOLD}       {i}. {opt['label']}{C.NC} {C.DIM}[{opt_color}{opt['type']}{C.NC}{C.DIM}]{C.NC}")
            print(f"  {C.DIM}       conf={confidence:.2f}  {latency:.0f}ms{C.NC}")
        else:
            action_hint = ''
            if resp_type.startswith('action_'):
                action_name = resp_type.replace('action_', '')
                action_hint = f"\n  {C.CYAN}       ⚡ App action: {action_name}{C.NC}"
            print(f"  {C.BOLD}  Bot:{C.NC} {resp_text}{action_hint}")
            print(f"  {C.DIM}       [{type_color}{resp_type}{C.NC}{C.DIM}] tag={tag}  conf={confidence:.2f}  {latency:.0f}ms{C.NC}")

        if debug:
            top_n = evaluator.predict_top_n(user_input, lang=lang, n=5)
            print(f"  {C.DIM}       scores: ", end='')
            parts = [f"{t}={p:.2f}" for t, p in top_n]
            print(f"{' | '.join(parts)}{C.NC}")

        print()


if __name__ == '__main__':
    main()
