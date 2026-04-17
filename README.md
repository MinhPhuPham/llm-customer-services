# On-Device Bilingual Customer Support AI

Intent classification (EN/JP) → train on Colab A100 → export CoreML (iOS) + TFLite (Android).

```
[EN]/[JA] + query → mmBERT-small (pruned EN+JP) → Intent → Answer Template
                                                    ↓        ↓        ↓
                                                  ≥0.85    <0.85   support
                                                 template  decline    form
```

| Spec | Value |
|------|-------|
| Model | mmBERT-small, 140M params, vocab pruned 256K→~35K |
| Training | 10 epochs, batch 2048, sqrt-scaled LR (1.6e-4), cosine schedule |
| iOS | CoreML FP16 (.mlpackage) — Apple Neural Engine |
| Android | TFLite INT8 (.tflite) — NNAPI/GPU delegate |
| On-device | ~50MB total, ~30-40MB RAM |
| Language | Explicit `[EN]`/`[JA]` prefix from device locale (not auto-detect) |

---

## Structure

```
scripts/
├── config.py                    # Constants, hyperparams, platform/GPU detection
├── setup.py                     # Drive mount, repo clone, output dirs
├── run_pipeline.py              # CLI — runs full pipeline end-to-end
├── data_loader/
│   ├── excel_parser.py          # Excel + Google Sheets → bilingual training pairs
│   ├── dataset_builder.py       # Tokenize + stratified split → HF Dataset
│   └── export_artifacts.py      # label_map.json + responses.json for mobile
└── helpers/
    ├── vocab_pruner.py          # Strip unused languages from embeddings
    ├── trainer_factory.py       # Classifier + HF Trainer (cosine LR, f1_macro)
    ├── training_plots.py        # Loss / accuracy / F1 charts
    ├── export_coreml.py         # → .mlpackage FP16
    ├── export_tflite.py         # → .tflite INT8
    ├── compress.py              # Zstd compression for OTA
    └── evaluator.py             # TFLite inference + EN/JA tests + latency

notebook/
└── train_customer_services_model.ipynb                  # Colab notebook — step-by-step with Google Sheets support
```

---

## Quick Start (Colab)

1. Open `notebook/train_customer_services_model.ipynb` in Colab
2. Fill in `GITHUB_TOKEN`, `REPO_OWNER`, `REPO_NAME`
3. Run all cells

**CLI alternative:**
```bash
python -m scripts.run_pipeline                                          # full pipeline
python -m scripts.run_pipeline --steps 1,2,3,4                         # train only
python -m scripts.run_pipeline --google-sheet "https://docs.google.com/spreadsheets/d/..."
```

---

## Training Data

Excel or Google Sheets with this layout:

| intent | type | Question EN | Question JA | Answer EN | Answer JA |
|--------|------|-------------|-------------|-----------|-----------|
| account_password | answer | How to reset? | パスワードをリセットしたい | To reset: Settings > ... | リセット：設定 > ... |
| account_password | answer | forgot password | パスワードを忘れました | *(inherits above)* | |

- `type`: `answer` = direct reply, `support` = show form, `reject` = out of scope
- Answers filled only on first row per intent, rest auto-inherited

---

## Export Artifacts

| File | Used by |
|------|---------|
| `SupportAI.mlpackage` | iOS (Xcode, On-Demand Resources) |
| `support_ai.tflite` | Android (Play Asset Delivery) |
| `support_ai.tflite.zst` | OTA delivery via CDN |
| `label_map.json` | Both — intent_id → intent_name |
| `responses.json` | Both — intent → {en, ja, type} |
