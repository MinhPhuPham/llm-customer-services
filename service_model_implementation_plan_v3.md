# On-Device Bilingual Customer Support AI

Intent Classification model (EN/JP) for mobile on-device customer support.
Train on server → export CoreML (iOS) + TFLite (Android) → deliver via platform-native asset systems.

## Architecture
```
[EN/JA] + user_query → Tokenizer → mmBERT-small (pruned EN+JP) → Intent → Answer Template
   ↑                                                                 ↓
language from                                                 confidence check
device locale                                                 ↓        ↓        ↓
(not auto-detect)                                          ≥0.85    <0.85   support
                                                          template  decline    form
```

| Component | Detail |
|-----------|--------|
| Base Model | mmBERT-small (140M params, ModernBERT arch, 2025) |
| Vocab Pruning | 256K → ~35K tokens (EN + JP only) |
| Language Input | Explicit `[EN]`/`[JA]` prefix from device locale |
| Training Data | Google Sheets / Excel — business team controls |
| Training | Google Colab A100, ~2-4h production |
| iOS Export | **CoreML** (.mlpackage, FP16) — Apple Neural Engine |
| Android Export | **TFLite** (.tflite, INT8) — NNAPI/GPU delegate |
| Delivery | Android: Play Asset Delivery / iOS: On-Demand Resources |
| App Store Size | ~12-15MB visible (model delivered separately) |
| On-Device Size | ~50-55MB total |
| Runtime RAM | ~30-40MB |

## Why explicit `[EN]`/`[JA]` prefix (not auto-detect)

| | Auto-detect | Explicit prefix |
|---|---|---|
| Accuracy | Wastes model capacity guessing | ~5-10% better intent accuracy |
| Speed | Extra cycles on ambiguous text | No detection overhead |
| Edge cases | "OK" / "WiFi" / "API" → ambiguous | Unambiguous |
| App code | Need detection library (~2MB) | One line: read device locale |

## Pipeline
```
1. Setup         — clone repo, mount Drive, detect A100
2. Config        — TESTING_MODE or PRODUCTION
3. Data Load     — read Excel from Drive → parse bilingual Q&A → add [EN]/[JA] prefix
4. Vocab Prune   — strip 102 unused languages from tokenizer + embeddings
5. Fine-tune     — classification head on A100 with cosine LR
6. Export iOS    — PyTorch → TorchScript → CoreML FP16
7. Export Android — PyTorch → TF SavedModel → TFLite INT8
8. Evaluate      — bilingual test cases with TFLite interpreter
9. Save          — all artifacts to Google Drive
10. Delivery     — Android AI Pack / iOS ODR / OTA via CDN
```

## Excel Template Layout (1 sheet, business team edits)
```
intent | type | Question         | Answer
       |      | English | Japanese | English | Japanese
───────┼──────┼─────────┼────────┼─────────┼────────
acct.. | ans  | How to  | パス.. | To reset| リセット
acct.. | ans  | forgot  | 忘れ.. |         |           ← answer blank = inherits from above
acct.. | ans  | locked  | ログ.. |         |
sub_.. | ans  | cancel  | 解約.. | To canc | キャンセル
...    |      |         |        |         |
```
- `type`: `answer` = direct reply, `support` = show form, `reject` = out of scope
- Answer filled only on first row per intent, rest auto-inherited
- EN/JA questions don't need 1:1 pairing — fill what you have


---

## 1. Setup

```python
# ===========================================================
# DEPENDENCIES
# ===========================================================
!pip install transformers>=4.51.0 accelerate peft -q
!pip install onnx onnxruntime optimum -q
!pip install coremltools tensorflow-cpu -q
!pip install datasets scikit-learn openpyxl -q
!pip install matplotlib tqdm zstandard -q

import os, sys, gc, json, subprocess, shutil
import numpy as np
import pandas as pd

# ===========================================================
# PLATFORM DETECTION
# ===========================================================
if os.path.exists('/kaggle/working'):
    PLATFORM = 'kaggle'
elif os.path.exists('/content'):
    PLATFORM = 'colab'
else:
    PLATFORM = 'local'
print(f"Platform: {PLATFORM}")

# ===========================================================
# REPO CONFIG (private GitHub)
# ===========================================================
REPO_BRANCH = 'main'
REPO_OWNER = '<YourGitHubUser>'
REPO_NAME = '<YourRepoName>'

GITHUB_TOKEN = ''
try:
    if PLATFORM == 'colab':
        from google.colab import userdata
        GITHUB_TOKEN = userdata.get('GITHUB_TOKEN') or ''
    elif PLATFORM == 'kaggle':
        from kaggle_secrets import UserSecretsClient
        GITHUB_TOKEN = UserSecretsClient().get_secret('GITHUB_TOKEN') or ''
except Exception:
    pass

if GITHUB_TOKEN:
    REPO_URL = f'https://{GITHUB_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git'
    print("  Private repo (token set)")
else:
    REPO_URL = f'https://github.com/{REPO_OWNER}/{REPO_NAME}.git'
    print("  Public repo (no token)")

# ===========================================================
# GOOGLE DRIVE + CLONE
# ===========================================================
if PLATFORM == 'colab':
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_DIR = '/content/SupportAI'
    DRIVE_DIR = '/content/drive/MyDrive/SupportAI'
elif PLATFORM == 'kaggle':
    REPO_DIR = '/kaggle/working/SupportAI'
    DRIVE_DIR = '/kaggle/working/output'
else:
    REPO_DIR = os.path.expanduser('~/SupportAI')
    DRIVE_DIR = REPO_DIR + '/output'

DATA_DIR = f'{DRIVE_DIR}/data'
MODEL_DIR = f'{DRIVE_DIR}/models'
EXPORT_DIR = f'{DRIVE_DIR}/export'
for d in [DATA_DIR, MODEL_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(REPO_DIR):
    !git clone -b {REPO_BRANCH} {REPO_URL} {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull origin {REPO_BRANCH}

REPO_HEAD = subprocess.check_output(
    ['git', '-C', REPO_DIR, 'rev-parse', '--short', 'HEAD'], text=True
).strip()
print(f"Repo: {REPO_HEAD} ({REPO_BRANCH})")
sys.path.insert(0, REPO_DIR)

# ===========================================================
# GPU
# ===========================================================
import torch
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_MEM = torch.cuda.get_device_properties(0).total_mem / 1e9
    DEVICE = 'cuda'
    DTYPE = torch.bfloat16 if 'A100' in GPU_NAME else torch.float16
    print(f"GPU: {GPU_NAME} ({GPU_MEM:.1f}GB), dtype={DTYPE}")
else:
    DEVICE = 'cpu'
    DTYPE = torch.float32
    print("WARNING: No GPU — will be slow")
```


---

## 2. Configuration

```python
# ===========================================================
# MODE
# ===========================================================
TESTING_MODE = True
CONTINUE_TRAINING = False

# ===========================================================
# BASE MODEL (mmBERT-small recommended, xlm-roberta-base as fallback)
# ===========================================================
BASE_MODEL = 'jhu-clsp/mmbert-small'

# ===========================================================
# LANGUAGE PREFIXES
# ===========================================================
LANG_PREFIX = {'en': '[EN]', 'ja': '[JA]'}

# ===========================================================
# TRAINING PARAMS
# ===========================================================
MAX_SEQ_LENGTH = 64
CONFIDENCE_THRESHOLD = 0.85

if TESTING_MODE:
    NUM_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    print(f"TESTING: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}")
else:
    NUM_EPOCHS = 10
    BATCH_SIZE = 64 if DEVICE == 'cuda' else 16
    LEARNING_RATE = 2e-5
    print(f"PRODUCTION: {NUM_EPOCHS} epochs, batch={BATCH_SIZE}, LR={LEARNING_RATE}")
```


---

## 3. Load Training Data from Excel

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ===========================================================
# READ EXCEL (side-by-side EN/JA layout)
# ===========================================================
EXCEL_PATH = f'{DATA_DIR}/training_data_template.xlsx'
assert os.path.exists(EXCEL_PATH), f"Excel not found: {EXCEL_PATH}"

df = pd.read_excel(EXCEL_PATH, header=[0, 1])
df.columns = ['intent', 'type', 'q_en', 'q_ja', 'a_en', 'a_ja']

# Forward-fill answers within each intent group
df['a_en'] = df.groupby('intent')['a_en'].ffill()
df['a_ja'] = df.groupby('intent')['a_ja'].ffill()

# ===========================================================
# FLATTEN TO TRAINING ROWS + ADD LANGUAGE PREFIX
# ===========================================================
train_rows = []
for _, row in df.iterrows():
    if pd.notna(row['q_en']) and str(row['q_en']).strip():
        train_rows.append(('[EN] ' + str(row['q_en']).strip(), row['intent']))
    if pd.notna(row['q_ja']) and str(row['q_ja']).strip():
        train_rows.append(('[JA] ' + str(row['q_ja']).strip(), row['intent']))

texts = [r[0] for r in train_rows]
labels = [r[1] for r in train_rows]

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(label_encoder.classes_)

en_count = sum(1 for t in texts if t.startswith('[EN]'))
ja_count = sum(1 for t in texts if t.startswith('[JA]'))
print(f"Samples: {len(texts)} ({en_count} EN + {ja_count} JA)")
print(f"Intents: {num_labels} — {list(label_encoder.classes_)}")

# ===========================================================
# TRAIN / VAL SPLIT
# ===========================================================
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, encoded_labels, test_size=0.15, stratify=encoded_labels, random_state=42
)
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

# ===========================================================
# SAVE ARTIFACTS FOR MOBILE APP
# ===========================================================
# 1. Label map: intent_id → intent_name
label_map = {int(i): lbl for i, lbl in enumerate(label_encoder.classes_)}
with open(f'{EXPORT_DIR}/label_map.json', 'w') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

# 2. Response templates: intent → {en: "...", ja: "..."}
responses = {}
for _, row in df.drop_duplicates('intent').iterrows():
    responses[row['intent']] = {
        'en': str(row['a_en']) if pd.notna(row['a_en']) else '',
        'ja': str(row['a_ja']) if pd.notna(row['a_ja']) else '',
    }
    # Preserve intent type (answer/support/reject) for app logic
    responses[row['intent']]['type'] = row['type']

with open(f'{EXPORT_DIR}/responses.json', 'w', encoding='utf-8') as f:
    json.dump(responses, f, ensure_ascii=False, indent=2)

print(f"Saved: label_map.json ({num_labels}), responses.json ({len(responses)} intents)")
```


---

## 4. Vocabulary Pruning (EN + JP only)

```python
from transformers import AutoTokenizer, AutoModel

# ===========================================================
# LOAD BASE MODEL
# ===========================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModel.from_pretrained(BASE_MODEL, torch_dtype=DTYPE)

original_vocab = tokenizer.vocab_size
original_params = sum(p.numel() for p in base_model.parameters())
print(f"Original: {original_vocab:,} vocab, {original_params:,} params")

# ===========================================================
# COLLECT USED TOKENS (training data + safety buffer)
# ===========================================================
used_tokens = set()

# From all training patterns
for text in texts:
    used_tokens.update(tokenizer.encode(text, add_special_tokens=False))

# All special tokens
for sid in tokenizer.all_special_ids:
    used_tokens.add(sid)

# Basic ASCII + common symbols
for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:\'"()-/@#$%& ':
    used_tokens.update(tokenizer.encode(char, add_special_tokens=False))

# Optional: broader coverage from Wikipedia (uncomment for production)
# from datasets import load_dataset
# for sample in load_dataset('wikipedia', '20220301.ja', split='train[:5000]'):
#     used_tokens.update(tokenizer.encode(sample['text'][:500], add_special_tokens=False))
# for sample in load_dataset('wikipedia', '20220301.en', split='train[:5000]'):
#     used_tokens.update(tokenizer.encode(sample['text'][:500], add_special_tokens=False))

kept_ids = sorted(used_tokens)
print(f"Pruned vocab: {len(kept_ids):,} ({len(kept_ids)/original_vocab*100:.1f}% of original)")

# ===========================================================
# REBUILD EMBEDDING MATRIX
# ===========================================================
old_emb = base_model.embeddings.word_embeddings.weight.data
new_emb = torch.nn.Embedding(len(kept_ids), old_emb.shape[1])
for new_id, old_id in enumerate(kept_ids):
    new_emb.weight.data[new_id] = old_emb[old_id]

base_model.embeddings.word_embeddings = new_emb
base_model.config.vocab_size = len(kept_ids)

# Save token ID mapping (needed for tokenizer reconstruction)
old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_ids)}
with open(f'{EXPORT_DIR}/token_id_map.json', 'w') as f:
    json.dump({str(k): v for k, v in old_to_new.items()}, f)

pruned_params = sum(p.numel() for p in base_model.parameters())
print(f"Pruned: {pruned_params:,} params ({(1-pruned_params/original_params)*100:.0f}% reduction)")
del old_emb; gc.collect()
```


---

## 5. Fine-tune

```python
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score

# ===========================================================
# BUILD CLASSIFIER
# ===========================================================
classifier = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=num_labels, torch_dtype=DTYPE,
)
# Apply pruned embeddings
classifier.base_model.embeddings.word_embeddings = base_model.embeddings.word_embeddings
classifier.config.vocab_size = len(kept_ids)
classifier.to(DEVICE)

print(f"Classifier: {sum(p.numel() for p in classifier.parameters()):,} params, "
      f"{num_labels} intents")

# ===========================================================
# TOKENIZE
# ===========================================================
def tokenize_fn(examples):
    return tokenizer(examples['text'], padding='max_length',
                     truncation=True, max_length=MAX_SEQ_LENGTH)

train_ds = Dataset.from_dict({'text': train_texts, 'label': train_labels.tolist()})
val_ds = Dataset.from_dict({'text': val_texts, 'label': val_labels.tolist()})
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['text']).with_format('torch')
val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=['text']).with_format('torch')

# ===========================================================
# TRAIN
# ===========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }

training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    greater_is_better=True,
    logging_steps=50,
    bf16=(DTYPE == torch.bfloat16),
    fp16=(DTYPE == torch.float16),
    dataloader_num_workers=2,
    report_to='none',
)

trainer = Trainer(
    model=classifier,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# Resume if requested
resume_path = None
if CONTINUE_TRAINING:
    ckpts = sorted([d for d in os.listdir(MODEL_DIR) if d.startswith('checkpoint-')],
                   key=lambda x: int(x.split('-')[-1]))
    if ckpts:
        resume_path = os.path.join(MODEL_DIR, ckpts[-1])
        print(f"Resuming from {resume_path}")

history = trainer.train(resume_from_checkpoint=resume_path)
print(f"Done: {history.metrics}")
```


---

## 6. Training Plots

```python
import matplotlib.pyplot as plt

log_history = trainer.state.log_history
train_loss = [x['loss'] for x in log_history if 'loss' in x and 'eval_loss' not in x]
eval_entries = [x for x in log_history if 'eval_loss' in x]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_loss, alpha=0.5, label='Train')
axes[0].plot(np.linspace(0, len(train_loss), len(eval_entries)),
             [x['eval_loss'] for x in eval_entries], 'r-o', label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot([x['eval_accuracy'] for x in eval_entries], 'g-o')
axes[1].set_title('Val Accuracy'); axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

axes[2].plot([x['eval_f1_macro'] for x in eval_entries], 'b-o')
axes[2].set_title('Val F1 (macro)'); axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{EXPORT_DIR}/training_plots.png', dpi=150)
plt.show()
print(f"Best accuracy: {max(x['eval_accuracy'] for x in eval_entries):.4f}")
```


---

## 7. Export CoreML (iOS) + TFLite (Android)

### 7.1 Save best PyTorch model

```python
BEST_DIR = f'{EXPORT_DIR}/best_pytorch'
trainer.save_model(BEST_DIR)
tokenizer.save_pretrained(BEST_DIR)
print(f"Saved PyTorch model to {BEST_DIR}")
```

### 7.2 CoreML (.mlpackage, FP16)

```python
import coremltools as ct

best_model = trainer.model.eval().cpu()

# Trace
dummy_ids = torch.randint(0, 100, (1, MAX_SEQ_LENGTH), dtype=torch.long)
dummy_mask = torch.ones(1, MAX_SEQ_LENGTH, dtype=torch.long)
traced = torch.jit.trace(best_model, (dummy_ids, dummy_mask), strict=False)

# Convert
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name='input_ids', shape=(1, MAX_SEQ_LENGTH), dtype=np.int32),
        ct.TensorType(name='attention_mask', shape=(1, MAX_SEQ_LENGTH), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name='logits')],
    convert_to='mlprogram',
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
)
mlmodel.author = 'SupportAI'
mlmodel.short_description = 'Bilingual EN/JP intent classifier'
mlmodel.version = '1.0'

coreml_path = f'{EXPORT_DIR}/SupportAI.mlpackage'
mlmodel.save(coreml_path)
coreml_size = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fn in os.walk(coreml_path) for f in fn) / 1e6
print(f"CoreML: {coreml_size:.1f}MB (FP16)")
```

### 7.3 TFLite (.tflite, INT8)

```python
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

# Load as TF model
tf_model = TFAutoModelForSequenceClassification.from_pretrained(BEST_DIR, from_pt=True)

# Concrete function with fixed shape
@tf.function(input_signature=[
    tf.TensorSpec([1, MAX_SEQ_LENGTH], tf.int32, name='input_ids'),
    tf.TensorSpec([1, MAX_SEQ_LENGTH], tf.int32, name='attention_mask'),
])
def serve(input_ids, attention_mask):
    return {'logits': tf_model(input_ids=input_ids, attention_mask=attention_mask).logits}

# Convert with INT8 dynamic range quantization
converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
tflite_model = converter.convert()

tflite_path = f'{EXPORT_DIR}/support_ai.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

tflite_size = os.path.getsize(tflite_path) / 1e6
print(f"TFLite: {tflite_size:.1f}MB (INT8)")
```

### 7.4 Compress for OTA delivery

```python
import zstandard as zstd

for name, src in [('TFLite', tflite_path)]:
    dst = src + '.zst'
    cctx = zstd.ZstdCompressor(level=19)
    with open(src, 'rb') as f:
        raw = f.read()
    with open(dst, 'wb') as f:
        f.write(cctx.compress(raw))
    print(f"  {name}: {len(raw)/1e6:.1f}MB → {os.path.getsize(dst)/1e6:.1f}MB (zstd)")
```


---

## 8. Evaluate

```python
# ===========================================================
# TFLITE INFERENCE (simulates mobile)
# ===========================================================
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
inp_details = interpreter.get_input_details()
out_details = interpreter.get_output_details()

def predict(text, lang='en', threshold=CONFIDENCE_THRESHOLD):
    full = f"{LANG_PREFIX[lang]} {text}"
    enc = tokenizer(full, return_tensors='np', padding='max_length',
                    truncation=True, max_length=MAX_SEQ_LENGTH)
    interpreter.set_tensor(inp_details[0]['index'], enc['input_ids'].astype(np.int32))
    interpreter.set_tensor(inp_details[1]['index'], enc['attention_mask'].astype(np.int32))
    interpreter.invoke()
    logits = interpreter.get_tensor(out_details[0]['index'])
    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    pid = np.argmax(probs, axis=-1)[0]
    conf = probs[0][pid]
    intent = label_map[int(pid)]
    return ('out_of_scope', conf) if conf < threshold else (intent, conf)

# ===========================================================
# VALIDATION ACCURACY (ONNX INT8)
# ===========================================================
correct = sum(1 for t, l in zip(val_texts, val_labels)
              if label_encoder.classes_[l] == predict(
                  t.replace('[EN] ', '').replace('[JA] ', ''),
                  lang='en' if t.startswith('[EN]') else 'ja',
                  threshold=0.0)[0])
print(f"Val accuracy (TFLite): {correct/len(val_texts)*100:.1f}%")

# ===========================================================
# ENGLISH TESTS
# ===========================================================
print('\n' + '=' * 60)
print('English Tests (lang="en")')
print('=' * 60)
for text, expected in [
    ("How do I reset my password?", "account_password"),
    ("Cancel my subscription", "subscription_cancel"),
    ("The app keeps crashing", "bug_report"),
    ("What's the weather?", "out_of_scope"),
    ("I need to talk to someone", "need_human"),
]:
    intent, conf = predict(text, lang='en')
    s = 'PASS' if intent == expected else 'FAIL'
    print(f"  [{s}] \"{text}\" → {intent} ({conf:.2f})")

# ===========================================================
# JAPANESE TESTS
# ===========================================================
print('\n' + '=' * 60)
print('Japanese Tests (lang="ja")')
print('=' * 60)
for text, expected in [
    ("パスワードを変更したい", "account_password"),
    ("解約したい", "subscription_cancel"),
    ("アプリがフリーズする", "bug_report"),
    ("今日の天気は？", "out_of_scope"),
    ("サポートに連絡したい", "need_human"),
]:
    intent, conf = predict(text, lang='ja')
    s = 'PASS' if intent == expected else 'FAIL'
    print(f"  [{s}] \"{text}\" → {intent} ({conf:.2f})")

# ===========================================================
# LATENCY
# ===========================================================
import time
times = []
for _ in range(100):
    t0 = time.perf_counter()
    predict("How do I reset my password?", lang='en')
    times.append((time.perf_counter() - t0) * 1000)
print(f"\nTFLite latency: {np.median(times):.1f}ms median (CPU)")
```


---

## 9. Save to Google Drive

```python
print("=" * 60)
print("EXPORT FILES (Google Drive)")
print("=" * 60)

files = {
    'CoreML (iOS)':  coreml_path,
    'TFLite (Android)': tflite_path,
    'TFLite compressed': tflite_path + '.zst',
    'label_map.json': f'{EXPORT_DIR}/label_map.json',
    'responses.json': f'{EXPORT_DIR}/responses.json',
    'token_id_map.json': f'{EXPORT_DIR}/token_id_map.json',
    'training_plots.png': f'{EXPORT_DIR}/training_plots.png',
}

total = 0
for name, path in files.items():
    if os.path.isdir(path):
        sz = sum(os.path.getsize(os.path.join(dp, f))
                 for dp, _, fn in os.walk(path) for f in fn) / 1e6
    elif os.path.exists(path):
        sz = os.path.getsize(path) / 1e6
    else:
        sz = 0
    total += sz
    print(f"  {name:25s} {sz:8.1f}MB")

print(f"  {'TOTAL':25s} {total:8.1f}MB")
print(f"\nSaved to: {EXPORT_DIR}")
```


---

## 10. Mobile Integration

### iOS (Swift + CoreML)

```swift
class SupportAI {
    private let model: SupportAIModel  // from SupportAI.mlpackage
    private let responses: [String: [String: String]]
    private let threshold: Float = 0.85

    func answer(query: String) -> SupportResponse {
        let lang = Locale.current.language.languageCode?.identifier == "ja" ? "ja" : "en"
        let input = (lang == "ja" ? "[JA] " : "[EN] ") + query
        let (intent, confidence) = classify(input)

        if confidence < threshold {
            return .outOfScope(responses["out_of_scope"]![lang]!)
        }
        let info = responses[intent]!
        if info["type"] == "support" {
            return .supportForm(info[lang]!)
        }
        return .answer(info[lang]!)
    }
}
```

### Android (Kotlin + TFLite)

```kotlin
class SupportAI(context: Context) {
    private val interpreter = Interpreter(loadModel(context, "support_ai.tflite"))
    private val responses: Map<String, Map<String, String>>
    private val threshold = 0.85f

    fun answer(query: String): SupportResponse {
        val lang = if (Locale.getDefault().language == "ja") "ja" else "en"
        val input = if (lang == "ja") "[JA] $query" else "[EN] $query"
        val (intent, confidence) = classify(input)

        if (confidence < threshold)
            return SupportResponse.OutOfScope(responses["out_of_scope"]!![lang]!!)
        val info = responses[intent]!!
        if (info["type"] == "support")
            return SupportResponse.Form(info[lang]!!)
        return SupportResponse.Answer(info[lang]!!)
    }
}
```


---

## 11. Delivery

### App Store / Play Store size

| Component | iOS | Android |
|-----------|-----|---------|
| App code + UI | ~8MB | ~8MB |
| AI model (via asset delivery) | ~25-35MB CoreML | ~15-25MB TFLite |
| responses.json (bundled) | ~3MB | ~3MB |
| **Visible in store** | **~12MB** | **~12MB** |
| **Total on device** | **~48MB** | **~48MB** |

### Android: Play Asset Delivery

```groovy
// ai-model-pack/build.gradle
plugins { id 'com.android.asset-pack' }
assetPack {
    packName = "ai-model-pack"
    dynamicDelivery { deliveryType = "fast-follow" }
}
```
Model downloads silently after app install. No user action needed.

### iOS: On-Demand Resources

In Xcode: select `.mlpackage` → File Inspector → On Demand Resource Tags: `ai-model` → Category: Initial Install.

### OTA Model Updates (no App Store review)

```
Your CDN (Firebase Storage / S3 / Cloudflare R2)
├── manifest.json          ← app checks this on launch
├── v1.0/
│   ├── support_ai.tflite.zst
│   └── SupportAI.mlpackage.zst
└── v1.1/                  ← retrained with new FAQ content
    ├── support_ai.tflite.zst
    └── SupportAI.mlpackage.zst
```

```json
{
  "version": "1.1",
  "tflite_url": "https://cdn.yourapp.com/v1.1/support_ai.tflite.zst",
  "coreml_url": "https://cdn.yourapp.com/v1.1/SupportAI.mlpackage.zst",
  "sha256_tflite": "abc123...",
  "sha256_coreml": "def456...",
  "min_app_version": "2.0.0"
}
```

App checks manifest on launch → downloads new model in background → atomic swap → done. Retrain with new Excel data and push to all users in minutes.
