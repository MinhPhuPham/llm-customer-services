"""End-to-end local test: 1-epoch train → CoreML → TFLite export."""
import sys, os, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    # Force testing mode + CPU
    import scripts.config as cfg
    cfg.TESTING_MODE = True
    cfg.DEVICE = 'cpu'
    cfg.DTYPE = __import__('torch').float32
    cfg.IS_HIGH_END_GPU = False
    cfg.NUM_EPOCHS = 1
    cfg.BATCH_SIZE = 4
    cfg.LEARNING_RATE = 1e-3

    import tempfile, shutil
    TMPDIR = tempfile.mkdtemp(prefix='e2e_test_')
    cfg.EXPORT_DIR = os.path.join(TMPDIR, 'export')
    cfg.MODEL_DIR = os.path.join(TMPDIR, 'models')
    os.makedirs(cfg.EXPORT_DIR, exist_ok=True)
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    print(f"Test output: {TMPDIR}")

    # ── Step 5: Load data from Excel ──
    print("\n=== STEP 5: Load Data ===")
    from scripts.data_loader.excel_parser import parse_excel
    from scripts.data_loader.dataset_builder import prepare_splits, tokenize_datasets
    from scripts.data_loader.export_artifacts import export_label_map, export_responses
    from transformers import AutoTokenizer

    excel_path = os.path.join(os.path.dirname(__file__), 'service_model_training_data_template.xlsx')
    if not os.path.exists(excel_path):
        print(f"SKIP: No Excel file at {excel_path}")
        print("Create a small test Excel or copy your training data.")
        sys.exit(0)

    train_rows, df = parse_excel(excel_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL)

    (texts, train_texts, val_texts,
     train_labels, val_labels,
     label_encoder, num_labels) = prepare_splits(train_rows)

    label_map = export_label_map(label_encoder)
    responses = export_responses(df)
    print(f"  {len(train_rows)} rows, {num_labels} tags")

    # ── Step 6: Vocab pruning + tokenization ──
    print("\n=== STEP 6: Vocab Pruning ===")
    from scripts.helpers.vocab_pruner import prune_vocabulary, RemappedTokenizer

    base_model, kept_ids, old_to_new = prune_vocabulary(tokenizer, texts)
    tokenizer = RemappedTokenizer(tokenizer, old_to_new)
    train_ds, val_ds = tokenize_datasets(
        train_texts, val_texts, train_labels, val_labels, tokenizer
    )

    # ── Step 7: Train 1 epoch ──
    print("\n=== STEP 7: Train (1 epoch) ===")
    from scripts.helpers.trainer_factory import build_trainer, run_training

    trainer, classifier = build_trainer(
        base_model, kept_ids, num_labels, train_ds, val_ds, tokenizer
    )
    history = run_training(trainer)

    # ── Step 8b: Save model for export ──
    print("\n=== STEP 8b: Save Model ===")
    from scripts.helpers.trainer_factory import save_best_model
    best_dir = save_best_model(trainer, tokenizer)

    # ── Step 9: CoreML export ──
    print("\n=== STEP 9: CoreML Export ===")
    from scripts.helpers.export_coreml import export_coreml

    coreml_path, coreml_size = export_coreml(tokenizer=tokenizer)
    print(f"  CoreML: {coreml_size:.1f}MB at {coreml_path}")

    # ── Step 10: TFLite export ──
    print("\n=== STEP 10: TFLite Export ===")
    from scripts.helpers.export_tflite import export_tflite

    tflite_path, tflite_size = export_tflite(
        tokenizer=tokenizer,
        calibration_texts=val_texts,
    )
    print(f"  TFLite: {tflite_size:.1f}MB at {tflite_path}")

    # ── Step 12: Evaluate ──
    print("\n=== STEP 12: Evaluate ===")
    from scripts.helpers.evaluator import evaluate_model

    results = evaluate_model(
        tflite_path, tokenizer, label_map, label_encoder,
        val_texts, val_labels,
    )

    # ── Cleanup ──
    print(f"\n=== DONE ===")
    print(f"  Accuracy: {results['accuracy']:.1f}%")
    print(f"  Latency:  {results['latency_ms']:.1f}ms")
    print(f"  Output:   {TMPDIR}")

    shutil.rmtree(TMPDIR, ignore_errors=True)
    print("  Cleaned up temp files")
    print("\n  ALL STEPS PASSED")


if __name__ == '__main__':
    multiprocessing.set_start_method('fork', force=True)
    main()
