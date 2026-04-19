# ===========================================================
# dataset_builder.py — Label encoding + split + tokenization
# ===========================================================
"""
Takes raw (text, tag) pairs and produces:
  - Encoded labels via LabelEncoder
  - Train/val splits (before pruning)
  - Tokenized HF Datasets (after pruning, with remapped tokenizer)

Split into two functions so that vocab pruning can happen between
label encoding and tokenization:

    prepare_splits()    →  raw text splits + label encoder
    prune_vocabulary()  →  pruned model + RemappedTokenizer
    tokenize_datasets() →  HF Datasets ready for Trainer
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

from scripts.config import MAX_SEQ_LENGTH


def prepare_splits(train_rows, test_size=0.15, random_state=42):
    """
    Encode labels and split into train/val sets (no tokenization).

    Call this BEFORE vocab pruning so that raw texts are available
    for token collection.  Tokenize AFTER pruning with tokenize_datasets().

    Args:
        train_rows: List of (prefixed_text, tag) tuples.
        test_size: Fraction for validation split.
        random_state: Random seed for reproducibility.

    Returns:
        texts: All text samples (needed by vocab pruner for token collection).
        train_texts: Training split texts.
        val_texts: Validation split texts.
        train_labels: Training encoded labels (numpy array).
        val_labels: Validation encoded labels (numpy array).
        label_encoder: Fitted LabelEncoder.
        num_labels: Number of unique tags.
    """
    texts = [r[0] for r in train_rows]
    labels = [r[1] for r in train_rows]

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    num_labels = len(label_encoder.classes_)

    print(f"  Tags: {num_labels} — {list(label_encoder.classes_)}")

    # Train / val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels,
        test_size=test_size,
        stratify=encoded_labels,
        random_state=random_state,
    )
    print(f"  Train: {len(train_texts)}, Val: {len(val_texts)}")

    return (
        texts, train_texts, val_texts,
        train_labels, val_labels,
        label_encoder, num_labels,
    )


def tokenize_datasets(train_texts, val_texts, train_labels, val_labels, tokenizer):
    """
    Tokenize pre-split data into HF Datasets ready for Trainer.

    Call this AFTER vocab pruning with a RemappedTokenizer so that
    token IDs match the pruned embedding matrix.

    Args:
        train_texts: Training split texts.
        val_texts: Validation split texts.
        train_labels: Training encoded labels (numpy array or list).
        val_labels: Validation encoded labels (numpy array or list).
        tokenizer: HuggingFace tokenizer (or RemappedTokenizer).

    Returns:
        train_ds: HF Dataset (tokenized, torch format).
        val_ds: HF Dataset (tokenized, torch format).
    """
    def tokenize_fn(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    train_labels_list = (
        train_labels.tolist() if hasattr(train_labels, 'tolist')
        else list(train_labels)
    )
    val_labels_list = (
        val_labels.tolist() if hasattr(val_labels, 'tolist')
        else list(val_labels)
    )

    train_ds = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels_list,
    })
    val_ds = Dataset.from_dict({
        'text': val_texts,
        'label': val_labels_list,
    })

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['text'])
    train_ds = train_ds.with_format('torch')

    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=['text'])
    val_ds = val_ds.with_format('torch')

    print(f"  Tokenized: {len(train_ds)} train, {len(val_ds)} val (seq_len={MAX_SEQ_LENGTH})")

    return train_ds, val_ds
