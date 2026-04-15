#!/usr/bin/env python3
# ===========================================================
# run_pipeline.py — CLI entry: runs full pipeline end-to-end
# ===========================================================
"""
Main orchestrator for the bilingual customer support AI pipeline.

Usage:
    python -m scripts.run_pipeline                    # Full pipeline
    python -m scripts.run_pipeline --steps 1,2,3      # Specific steps only
    python -m scripts.run_pipeline --skip-export       # Train only, skip export

Steps:
    1. Setup       — platform detection, dirs, Drive mount
    2. Data Load   — parse Excel, build datasets, export artifacts
    3. Vocab Prune — strip unused language tokens from embeddings
    4. Train       — fine-tune classifier with cosine LR
    5. Plot        — generate training history charts
    6. Export iOS   — CoreML .mlpackage (FP16)
    7. Export Android — TFLite .tflite (INT8)
    8. Compress    — zstd compression for OTA
    9. Evaluate    — bilingual test cases + latency
   10. Summary     — print all export file sizes
"""

import argparse
import os
import sys

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bilingual Customer Support AI — Training Pipeline'
    )
    parser.add_argument(
        '--steps', type=str, default=None,
        help='Comma-separated step numbers to run (e.g., "1,2,3,4"). Default: all.',
    )
    parser.add_argument(
        '--skip-export', action='store_true',
        help='Skip export steps (6-8). Useful for training-only runs.',
    )
    parser.add_argument(
        '--skip-eval', action='store_true',
        help='Skip evaluation step (9).',
    )
    parser.add_argument(
        '--excel', type=str, default=None,
        help='Path to training data Excel file. Default: auto-detect.',
    )
    parser.add_argument(
        '--google-sheet', type=str, default=None,
        help='Google Sheets URL or name. If set, loads from Sheets instead of Excel.',
    )
    return parser.parse_args()


def resolve_excel_path(cli_path=None):
    """Find the Excel training data file."""
    from scripts.config import DATA_DIR

    if cli_path and os.path.exists(cli_path):
        return cli_path

    # Check data dir
    data_path = os.path.join(DATA_DIR, 'training_data_template.xlsx')
    if os.path.exists(data_path):
        return data_path

    # Check project root
    root_path = os.path.join(PROJECT_ROOT, 'service_model_training_data_template.xlsx')
    if os.path.exists(root_path):
        return root_path

    raise FileNotFoundError(
        f"Excel file not found. Checked:\n"
        f"  1. CLI --excel arg\n"
        f"  2. {data_path}\n"
        f"  3. {root_path}\n"
        f"Place training_data_template.xlsx in one of these locations."
    )


def should_run(step_num, selected_steps, skip_export, skip_eval):
    """Check if a step should run based on CLI args."""
    if skip_export and step_num in (6, 7, 8):
        return False
    if skip_eval and step_num == 9:
        return False
    if selected_steps is not None:
        return step_num in selected_steps
    return True


def main():
    args = parse_args()

    # Parse selected steps
    selected_steps = None
    if args.steps:
        selected_steps = set(int(s.strip()) for s in args.steps.split(','))

    # ----------------------------------------------------------
    # Step 1: Setup
    # ----------------------------------------------------------
    if should_run(1, selected_steps, args.skip_export, args.skip_eval):
        from scripts.config import print_config
        from scripts.setup import run_setup
        print_config()
        run_setup()

    # Shared pipeline state (set by earlier steps, used by later ones)
    tokenizer = None
    val_texts = None

    # ----------------------------------------------------------
    # Step 2: Data Load (no tokenization — that happens after pruning)
    # ----------------------------------------------------------
    if should_run(2, selected_steps, args.skip_export, args.skip_eval):
        from scripts.config import EXPORT_DIR
        from scripts.data_loader.excel_parser import parse_excel, parse_google_sheet
        from scripts.data_loader.dataset_builder import prepare_splits
        from scripts.data_loader.export_artifacts import (
            export_label_map, export_responses,
        )
        from transformers import AutoTokenizer
        from scripts.config import BASE_MODEL

        print("=" * 60)
        print("DATA LOADING")
        print("=" * 60)

        # Load from Google Sheets or Excel
        google_sheet = getattr(args, 'google_sheet', None)
        if google_sheet:
            if google_sheet.startswith('http'):
                train_rows, df = parse_google_sheet(sheet_url=google_sheet)
            else:
                train_rows, df = parse_google_sheet(sheet_name=google_sheet)
        else:
            excel_path = resolve_excel_path(args.excel)
            print(f"  Excel: {excel_path}")
            train_rows, df = parse_excel(excel_path)

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        (
            texts, train_texts, val_texts,
            train_labels, val_labels,
            label_encoder, num_labels,
        ) = prepare_splits(train_rows)

        label_map = export_label_map(label_encoder)
        responses = export_responses(df)
        print()

    # ----------------------------------------------------------
    # Step 3: Vocab Prune + Remap Tokenizer + Tokenize
    # ----------------------------------------------------------
    if should_run(3, selected_steps, args.skip_export, args.skip_eval):
        from scripts.helpers.vocab_pruner import prune_vocabulary, RemappedTokenizer
        from scripts.data_loader.dataset_builder import tokenize_datasets
        base_model, kept_ids, old_to_new = prune_vocabulary(tokenizer, texts)
        tokenizer = RemappedTokenizer(tokenizer, old_to_new)
        train_ds, val_ds = tokenize_datasets(
            train_texts, val_texts, train_labels, val_labels, tokenizer
        )
        print()

    # ----------------------------------------------------------
    # Step 4: Train
    # ----------------------------------------------------------
    if should_run(4, selected_steps, args.skip_export, args.skip_eval):
        from scripts.helpers.trainer_factory import build_trainer, run_training
        trainer, classifier = build_trainer(
            base_model, kept_ids, num_labels, train_ds, val_ds, tokenizer
        )
        history = run_training(trainer)
        print()

    # ----------------------------------------------------------
    # Step 5: Plot
    # ----------------------------------------------------------
    if should_run(5, selected_steps, args.skip_export, args.skip_eval):
        from scripts.helpers.training_plots import plot_training_history
        print("=" * 60)
        print("TRAINING PLOTS")
        print("=" * 60)
        best_accuracy = plot_training_history(trainer)
        print()

    # ----------------------------------------------------------
    # Step 6: Export CoreML (iOS)
    # ----------------------------------------------------------
    coreml_path = None
    if should_run(6, selected_steps, args.skip_export, args.skip_eval):
        from scripts.helpers.export_coreml import export_coreml
        coreml_path, coreml_size = export_coreml(trainer, tokenizer)
        print()

    # ----------------------------------------------------------
    # Step 7: Export TFLite (Android)
    # ----------------------------------------------------------
    tflite_path = None
    if should_run(7, selected_steps, args.skip_export, args.skip_eval):
        from scripts.helpers.export_tflite import export_tflite
        tflite_path, tflite_size = export_tflite(
            tokenizer=tokenizer,
            calibration_texts=val_texts,
        )
        print()

    # ----------------------------------------------------------
    # Step 8: Compress for OTA
    # ----------------------------------------------------------
    if should_run(8, selected_steps, args.skip_export, args.skip_eval):
        from scripts.helpers.compress import compress_for_ota
        compress_files = []
        if tflite_path:
            compress_files.append(('TFLite', tflite_path))
        if compress_files:
            compress_for_ota(compress_files)
            print()

    # ----------------------------------------------------------
    # Step 9: Evaluate
    # ----------------------------------------------------------
    if should_run(9, selected_steps, args.skip_export, args.skip_eval):
        from scripts.helpers.evaluator import evaluate_model
        if tflite_path is None:
            from scripts.config import EXPORT_DIR
            tflite_path = os.path.join(EXPORT_DIR, 'support_ai.tflite')
        results = evaluate_model(
            tflite_path, tokenizer, label_map, label_encoder,
            val_texts, val_labels,
        )
        print()

    # ----------------------------------------------------------
    # Step 10: Summary
    # ----------------------------------------------------------
    if should_run(10, selected_steps, args.skip_export, args.skip_eval):
        from scripts.config import EXPORT_DIR
        print("=" * 60)
        print("EXPORT SUMMARY")
        print("=" * 60)

        export_files = {
            'CoreML (iOS)': os.path.join(EXPORT_DIR, 'SupportAI.mlpackage'),
            'TFLite (Android)': os.path.join(EXPORT_DIR, 'support_ai.tflite'),
            'TFLite compressed': os.path.join(EXPORT_DIR, 'support_ai.tflite.zst'),
            'label_map.json': os.path.join(EXPORT_DIR, 'label_map.json'),
            'responses.json': os.path.join(EXPORT_DIR, 'responses.json'),
            'token_id_map.json': os.path.join(EXPORT_DIR, 'token_id_map.json'),
            'training_plots.png': os.path.join(EXPORT_DIR, 'training_plots.png'),
        }

        total = 0
        for name, path in export_files.items():
            if os.path.isdir(path):
                sz = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fn in os.walk(path) for f in fn
                ) / 1e6
            elif os.path.exists(path):
                sz = os.path.getsize(path) / 1e6
            else:
                sz = 0
            total += sz
            print(f"  {name:25s} {sz:8.1f}MB")

        print(f"  {'TOTAL':25s} {total:8.1f}MB")
        print(f"\n  Saved to: {EXPORT_DIR}")
        print()

    print("✅ Pipeline complete!")


if __name__ == '__main__':
    main()
