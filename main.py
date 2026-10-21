"""
Transformer Encoder News Classifier (Pure Functional Pipeline)
==============================================================

This module implements the complete news classification pipeline using
self-documenting, descriptive function names in a purely functional style
without any custom OOP classes or nested functions.

Structure:
1. Imports and Global Configurations
2. Top-Level Helper Functions (Descriptive utility functions)
3. Data Preparation and Loading Functions
4. Model Layer Composition and Forward Pass Functions
5. Training and Evaluation Pipeline Functions
6. Prediction and Inference Functions
7. Architectural Verification (Smoke Test)
8. Command-Line Argument Parser and Main Entry Point
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

# Windows terminal UTF-8 output encoding fix
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# ============================================================================
# 1. Constants and Global Configurations
# ============================================================================

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_model.pt"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"

_TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", re.IGNORECASE)


# ============================================================================
# 2. Top-Level Helper Functions
# ============================================================================

def tokenize_english_text(text):
    return _TOKEN_PATTERN.findall(text.lower())


def encode_text_to_token_ids(text, vocab, max_len):
    tokens = tokenize_english_text(text)
    ids = []

    for token in tokens[:max_len]:
        if token in vocab:
            ids.append(vocab[token])
        else:
            ids.append(UNK_IDX)

    if len(ids) == 0:
        ids.append(UNK_IDX)

    return ids


def build_vocabulary_from_texts(texts, max_vocab_size=30000, min_freq=1):
    counter = Counter()

    for text in texts:
        tokens = tokenize_english_text(text)

        for token in tokens:
            counter[token] += 1

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}

    for token, count in counter.most_common():
        if count < min_freq:
            continue

        if len(vocab) >= max_vocab_size:
            break

        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def pad_and_collate_batch(batch):
    sequences = []
    labels = []

    for sequence, label in batch:
        sequences.append(sequence)
        labels.append(label)

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=PAD_IDX
    )

    label_tensor = torch.tensor(labels, dtype=torch.long)

    return padded, label_tensor


def set_random_seeds_for_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sinusoidal_positional_encodings(embedding_dim, max_len=512):
    if embedding_dim <= 0 or max_len <= 0:
        raise ValueError("embedding_dim and max_len must be positive integers.")

    positions = torch.arange(max_len).float().unsqueeze(1)
    pe = torch.zeros(max_len, embedding_dim)

    for pos in range(max_len):
        for i in range(embedding_dim):
            angle = pos / (10000 ** (2 * (i // 2) / embedding_dim))

            if i % 2 == 0:
                pe[pos, i] = math.sin(angle)
            else:
                pe[pos, i] = math.cos(angle)

    return pe.unsqueeze(0)


def plot_and_save_training_curves(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["valid_loss"], label="Validation Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(epochs, history["valid_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Training curves saved to: {path}")


def load_saved_model_checkpoint(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    return checkpoint


# ============================================================================
# 3. Data Preparation and Loading Functions
# ============================================================================

def prepare_encoded_news_dataset(split, vocab, max_len=128):
    dataset = []

    for row in split:
        text = str(row["text"])
        label = int(row["label"])

        ids = encode_text_to_token_ids(text, vocab, max_len)
        tensor = torch.tensor(ids, dtype=torch.long)

        dataset.append((tensor, label))

    return dataset


def download_and_split_ag_news_dataset(max_train_samples=12000, max_test_samples=2000, seed=42):
    from datasets import load_dataset

    print("Downloading/loading AG_NEWS dataset...")
    raw = load_dataset("fancyzhx/ag_news")

    train_source = raw["train"].shuffle(seed=seed)
    test_source = raw["test"].shuffle(seed=seed)

    if max_train_samples > 0:
        size = min(max_train_samples, len(train_source))
        train_source = train_source.select(range(size))

    if max_test_samples > 0:
        size = min(max_test_samples, len(test_source))
        test_source = test_source.select(range(size))

    split = train_source.train_test_split(test_size=0.05, seed=seed)

    train_split = split["train"]
    valid_split = split["test"]

    return train_split, valid_split, test_source


def create_pytorch_dataloaders(train_data, valid_data, test_data, batch_size=64, num_workers=0):
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_and_collate_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_and_collate_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_and_collate_batch,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, valid_loader, test_loader


# ============================================================================
# 4. Model Layer Composition and Forward Pass Functions
# ============================================================================

def build_transformer_classifier_layers(
    vocab_size,
    num_classes=4,
    embedding_dim=100,
    nhead=5,
    num_layers=2,
    dropout=0.1,
    pad_idx=0
):
    if embedding_dim % nhead != 0:
        raise ValueError("embedding_dim must be divisible by nhead")

    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embedding_dim,
        nhead=nhead,
        dim_feedforward=embedding_dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True
    )

    layers = nn.ModuleDict({
        "embedding": nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx),
        "dropout": nn.Dropout(dropout),
        "transformer_encoder": nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        ),
        "classifier": nn.Linear(embedding_dim, num_classes)
    })

    return layers


def forward_pass_transformer_classifier(
    layers,
    pe_table,
    token_ids,
    pad_idx=0,
    embedding_dim=100
):
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have 2 dimensions")

    padding_mask = token_ids.eq(pad_idx)

    x = layers["embedding"](token_ids) * math.sqrt(embedding_dim)

    seq_len = x.size(1)

    if seq_len > pe_table.size(1):
        raise ValueError("Sequence length exceeds positional encoding length")

    x = x + pe_table[:, :seq_len, :].to(
        device=x.device,
        dtype=x.dtype
    )

    x = layers["dropout"](x)

    x = layers["transformer_encoder"](
        x,
        src_key_padding_mask=padding_mask
    )

    valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)

    summed = (x * valid_mask).sum(dim=1)
    lengths = valid_mask.sum(dim=1).clamp(min=1.0)

    pooled = summed / lengths

    output = layers["dropout"](pooled)
    output = layers["classifier"](output)

    return output

# ============================================================================
# 5. Training and Evaluation Pipeline Functions
# ============================================================================

def execute_train_or_eval_epoch(
    layers,
    pe_table,
    loader,
    criterion,
    device,
    optimizer=None,
    pad_idx=0,
    embedding_dim=100
):
    training = optimizer is not None
    layers.train(training)

    total_loss = 0
    total_correct = 0
    total_items = 0

    for token_ids, labels in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = forward_pass_transformer_classifier(
                layers, pe_table, token_ids, pad_idx, embedding_dim
            )

            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(layers.parameters(), 1.0)
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += batch_size

    avg_loss = total_loss / max(total_items, 1)
    avg_acc = total_correct / max(total_items, 1)

    return avg_loss, avg_acc


def run_full_training_and_evaluation_pipeline(args):
    if args.epochs < 1 or args.max_len < 1:
        raise ValueError("epochs and max_len must be at least 1")

    set_random_seeds_for_reproducibility(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCompute Device: {device}")

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_split, valid_split, test_split = download_and_split_ag_news_dataset(
        args.max_train_samples,
        args.max_test_samples,
        args.seed
    )

    print(
        f"Train={len(train_split)}, "
        f"Valid={len(valid_split)}, Test={len(test_split)}"
    )

    vocab = build_vocabulary_from_texts(
        (str(row["text"]) for row in train_split),
        max_vocab_size=args.max_vocab_size
    )

    train_data = prepare_encoded_news_dataset(train_split, vocab, args.max_len)
    valid_data = prepare_encoded_news_dataset(valid_split, vocab, args.max_len)
    test_data = prepare_encoded_news_dataset(test_split, vocab, args.max_len)

    train_loader, valid_loader, test_loader = create_pytorch_dataloaders(
        train_data, valid_data, test_data, args.batch_size
    )

    layers = build_transformer_classifier_layers(
        vocab_size=len(vocab),
        num_classes=len(LABEL_NAMES),
        embedding_dim=args.embedding_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=PAD_IDX
    ).to(device)

    pe_table = generate_sinusoidal_positional_encodings(
        args.embedding_dim, args.max_len
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        layers.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.7
    )

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
        "learning_rate": []
    }

    best_valid_accuracy = -1
    checkpoint_path = checkpoint_dir / "best_model.pt"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = execute_train_or_eval_epoch(
            layers, pe_table, train_loader, criterion, device,
            optimizer, PAD_IDX, args.embedding_dim
        )

        valid_loss, valid_acc = execute_train_or_eval_epoch(
            layers, pe_table, valid_loader, criterion, device,
            None, PAD_IDX, args.embedding_dim
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_accuracy"].append(train_acc)
        history["valid_accuracy"].append(valid_acc)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss={train_loss:.4f}, Acc={train_acc:.2%} | "
            f"Valid Loss={valid_loss:.4f}, Acc={valid_acc:.2%} | "
            f"Time={time.time() - epoch_start:.1f}s"
        )

        if valid_acc > best_valid_accuracy:
            best_valid_accuracy = valid_acc

            checkpoint_payload = {
                "model_state_dict": layers.state_dict(),
                "vocab": vocab,
                "labels": LABEL_NAMES,
                "config": {
                    "vocab_size": len(vocab),
                    "num_classes": len(LABEL_NAMES),
                    "embedding_dim": args.embedding_dim,
                    "nhead": args.nhead,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "pad_idx": PAD_IDX,
                    "max_len": args.max_len,
                    "max_vocab_size": args.max_vocab_size
                },
                "best_valid_accuracy": best_valid_accuracy
            }

            torch.save(checkpoint_payload, checkpoint_path)
            print(f"Saved best model: {checkpoint_path}")

    plot_and_save_training_curves(
        history, results_dir / "training_curves.png"
    )

    with (results_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    best_ckpt = load_saved_model_checkpoint(checkpoint_path, device)
    layers.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc = execute_train_or_eval_epoch(
        layers, pe_table, test_loader, criterion, device,
        None, PAD_IDX, args.embedding_dim
    )

    elapsed = time.time() - start_time

    print("\nFinal Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Duration: {elapsed / 60:.2f} minutes")
    print(f"Checkpoint: {checkpoint_path}")

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_validation_accuracy": best_valid_accuracy,
        "training_minutes": elapsed / 60,
        "train_samples": len(train_data),
        "validation_samples": len(valid_data),
        "test_samples": len(test_data),
        "vocab_size": len(vocab)
    }

    with (results_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# ============================================================================
# 6. Prediction and Inference Functions
# ============================================================================

def predict_news_category_with_probabilities(
    text,
    checkpoint_path=DEFAULT_CHECKPOINT_PATH,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_saved_model_checkpoint(checkpoint_path, device)

    vocab = checkpoint["vocab"]
    labels = checkpoint["labels"]
    config = checkpoint["config"]

    layers = build_transformer_classifier_layers(
        vocab_size=config["vocab_size"],
        num_classes=config["num_classes"],
        embedding_dim=config["embedding_dim"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pad_idx=config["pad_idx"]
    ).to(device)

    layers.load_state_dict(checkpoint["model_state_dict"])
    layers.eval()

    pe_table = generate_sinusoidal_positional_encodings(
        config["embedding_dim"],
        config["max_len"]
    ).to(device)

    ids = encode_text_to_token_ids(text, vocab, config["max_len"])
    token_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = forward_pass_transformer_classifier(
            layers,
            pe_table,
            token_ids,
            pad_idx=config["pad_idx"],
            embedding_dim=config["embedding_dim"]
        )

        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_idx = probabilities.argmax().item()
    sorted_indices = probabilities.argsort(descending=True).tolist()

    scores = {}

    for idx in sorted_indices:
        scores[labels[idx]] = probabilities[idx].item()

    print("\n--- Prediction Result ---")
    print(f"Text: {text}")
    print(f"Predicted Category: {labels[predicted_idx]}")
    print("Class Probabilities:")

    for category, score in scores.items():
        bar = "=" * int(score * 25)
        print(f"{category}: {score:.2%} [{bar}]")

    print("-------------------------\n")

    return {
        "text": text,
        "predicted_label": labels[predicted_idx],
        "scores": scores
    }


# ============================================================================
# 7. Architectural Verification (Smoke Test)
# ============================================================================

def run_model_architecture_smoke_test():
    print("\n[Running Functional Smoke Test]...")

    torch.manual_seed(42)

    vocab_size = 500
    batch_size = 4
    sequence_length = 12
    embedding_dim = 100
    max_len = 128

    layers = build_transformer_classifier_layers(
        vocab_size,
        num_classes=4,
        embedding_dim=embedding_dim,
        nhead=5,
        num_layers=2,
        dropout=0.1,
        pad_idx=0
    )

    layers.eval()

    pe_table = generate_sinusoidal_positional_encodings(
        embedding_dim, max_len
    )

    fake_token_ids = torch.randint(
        1, vocab_size, (batch_size, sequence_length)
    )

    fake_token_ids[0, -3:] = 0
    fake_token_ids[1, -5:] = 0

    logits = forward_pass_transformer_classifier(
        layers,
        pe_table,
        fake_token_ids,
        pad_idx=0,
        embedding_dim=embedding_dim
    )

    assert logits.shape == (batch_size, 4)
    assert torch.isfinite(logits).all()

    print("Smoke test passed successfully.")
    print(f"Input shape: {tuple(fake_token_ids.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print("Transformer pipeline is verified.\n")

    return True


# ============================================================================
# 8. Command-Line Argument Parser and Main Entry Point
# ============================================================================


def ask_int(prompt, default, minimum=0):
    while True:
        value = input(f"{prompt} [default: {default}]: ").strip()

        if value == "":
            return default

        try:
            number = int(value)

            if number < minimum:
                print(f"Please enter a number >= {minimum}.")
                continue

            return number

        except ValueError:
            print("Invalid input. Enter an integer.")


def ask_float(prompt, default, minimum=0.0, minimum_exclusive=False):
    while True:
        value = input(f"{prompt} [default: {default}]: ").strip()

        if value == "":
            return default

        try:
            number = float(value)

            if not math.isfinite(number):
                print("Please enter a finite number.")
                continue

            if minimum_exclusive and number <= minimum:
                print(f"Please enter a number > {minimum}.")
                continue

            if not minimum_exclusive and number < minimum:
                print(f"Please enter a number >= {minimum}.")
                continue

            return number

        except ValueError:
            print("Invalid input. Enter a number.")


def get_training_arguments():
    print("\n========== Training Settings ==========")
    print("Press Enter to use the default value.")
    print("Enter 0 to use all available samples.")

    epochs = ask_int("Number of epochs", 20, 1)
    batch_size = ask_int("Batch size", 64, 1)
    learning_rate = ask_float("Learning rate", 0.001, 0.0, True)
    weight_decay = ask_float("Weight decay", 0.01, 0.0)
    max_train_samples = ask_int("Maximum training samples", 12000, 0)
    max_test_samples = ask_int("Maximum testing samples", 2000, 0)

    args = argparse.Namespace(
        mode="train",
        text=None,
        checkpoint=str(DEFAULT_CHECKPOINT_PATH),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_train_samples=max_train_samples,
        max_test_samples=max_test_samples,
        max_len=128,
        max_vocab_size=30000,
        embedding_dim=100,
        nhead=5,
        num_layers=2,
        dropout=0.1,
        seed=42,
        output_dir=str(BASE_DIR)
    )

    return args


def main():
    while True:
        print("\n==========================================")
        print("       TRANSFORMER NEWS CLASSIFIER")
        print("==========================================")
        print("1. Train the model")
        print("2. Predict a news category")
        print("3. Test model architecture")
        print("4. Exit")
        print("==========================================")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == "1":
            args = get_training_arguments()
            print("\nStarting training...\n")
            run_full_training_and_evaluation_pipeline(args)

        elif choice == "2":
            text = input("Enter news text in English:\n").strip()

            if not text:
                print("No text entered.")
                continue

            checkpoint_input = input(
                f"Checkpoint path [default: {DEFAULT_CHECKPOINT_PATH}]: "
            ).strip()

            checkpoint_path = Path(checkpoint_input) if checkpoint_input else DEFAULT_CHECKPOINT_PATH

            try:
                predict_news_category_with_probabilities(
                    text, checkpoint_path
                )
            except FileNotFoundError as exc:
                print(f"Error: {exc}")

        elif choice == "3":
            run_model_architecture_smoke_test()

        elif choice == "4":
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice. Enter a number from 1 to 4.")
            continue

        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()
    