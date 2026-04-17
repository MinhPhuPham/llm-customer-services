# ===========================================================
# trainer_factory.py — Build classifier + HF Trainer
# ===========================================================
"""
Creates the classification model and HuggingFace Trainer
with cosine LR schedule.
"""

import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from scripts.config import (
    BASE_MODEL, DEVICE, DTYPE,
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MODEL_DIR, CONTINUE_TRAINING,
)


def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }


def build_trainer(base_model, kept_ids, num_labels, train_ds, val_ds, tokenizer):
    """
    Build the classification model and HF Trainer.

    Args:
        base_model: Model with pruned embeddings.
        kept_ids: List of kept token IDs (for vocab size).
        num_labels: Number of intent classes.
        train_ds: Tokenized training HF Dataset.
        val_ds: Tokenized validation HF Dataset.
        tokenizer: HuggingFace tokenizer.

    Returns:
        trainer: Configured HF Trainer.
        classifier: The classification model.
    """
    print("=" * 60)
    print("BUILDING CLASSIFIER")
    print("=" * 60)

    # Build classifier from base config
    classifier = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=num_labels,
        dtype=DTYPE,
    )

    # Apply pruned embeddings (ModernBERT uses 'tok_embeddings', BERT uses 'word_embeddings')
    if hasattr(base_model.embeddings, 'tok_embeddings'):
        classifier.base_model.embeddings.tok_embeddings = base_model.embeddings.tok_embeddings
    else:
        classifier.base_model.embeddings.word_embeddings = base_model.embeddings.word_embeddings
    classifier.config.vocab_size = len(kept_ids)

    # Freeze pretrained layers — only train the classifier head.
    # mmBERT already understands EN+JA; with few samples (<500),
    # fine-tuning all 42M params causes severe overfitting.
    for param in classifier.base_model.parameters():
        param.requires_grad = False

    classifier.to(DEVICE)

    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"  Total: {total_params:,} params, {num_labels} intents")
    print(f"  Trainable: {trainable_params:,} (classifier head)")
    print(f"  Frozen: {frozen_params:,} (pretrained base)")

    # Training arguments — higher LR for head-only training
    head_lr = 1e-3

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=min(BATCH_SIZE * 2, 4096),
        learning_rate=head_lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='f1_macro',
        greater_is_better=True,
        logging_steps=1,
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

    print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {head_lr}")
    return trainer, classifier


def run_training(trainer):
    """
    Run the training loop, optionally resuming from checkpoint.

    Args:
        trainer: Configured HF Trainer.

    Returns:
        history: Training history metrics.
    """
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    resume_path = None
    if CONTINUE_TRAINING:
        ckpts = sorted(
            [d for d in os.listdir(MODEL_DIR) if d.startswith('checkpoint-')],
            key=lambda x: int(x.split('-')[-1]),
        )
        if ckpts:
            resume_path = os.path.join(MODEL_DIR, ckpts[-1])
            print(f"  Resuming from {resume_path}")

    history = trainer.train(resume_from_checkpoint=resume_path)
    print(f"  Done: {history.metrics}")
    return history
