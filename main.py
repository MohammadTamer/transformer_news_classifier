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

def tokenize_english_text(text: str) -> list[str]:
    """Tokenize an English text string into lowercase words and punctuation tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


def encode_text_to_token_ids(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    """Convert a text string into a list of integer token IDs bounded by max_len."""
    tokens = tokenize_english_text(text)[:max_len]
    ids = [vocab.get(token, UNK_IDX) for token in tokens]
    return ids if ids else [UNK_IDX]


def build_vocabulary_from_texts(
    texts: Iterable[str],
    max_vocab_size: int = 30000,
    min_freq: int = 1,
) -> dict[str, int]:
    """Build a word-to-index mapping dictionary strictly from training texts."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize_english_text(text))

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    available = max(0, max_vocab_size - len(vocab))

    for token, count in counter.most_common():
        if count < min_freq or len(vocab) - 2 >= available:
            break
        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def pad_and_collate_batch(batch: Sequence[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequence tensors to the longest sequence in the batch."""
    sequences, labels = zip(*batch)
    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, label_tensor


def set_random_seeds_for_reproducibility(seed: int = 42) -> None:
    """Set random seeds across Python random, NumPy, and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_sinusoidal_positional_encodings(embedding_dim: int, max_len: int = 512) -> torch.Tensor:
    """Compute sinusoidal position embedding matrix: [1, max_len, embedding_dim]."""
    if embedding_dim <= 0 or max_len <= 0:
        raise ValueError("embedding_dim and max_len must be positive integers.")

    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / embedding_dim)
    )

    pe = torch.zeros(max_len, embedding_dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


def plot_and_save_training_curves(history: dict[str, list[float]], path: Path) -> None:
    """Plot and save training and validation loss and accuracy progression curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs, history["valid_loss"], label="Validation Loss", marker="s")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy", marker="o")
    axes[1].plot(epochs, history["valid_accuracy"], label="Validation Accuracy", marker="s")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {path}")


def load_saved_model_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    """Safely load model weights and configuration checkpoint from disk."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            f"Train the model first using: python main.py --mode train"
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


# ============================================================================
# 3. Data Preparation and Loading Functions
# ============================================================================

def prepare_encoded_news_dataset(split, vocab: dict[str, int], max_len: int = 128) -> list[tuple[torch.Tensor, int]]:
    """Convert raw dataset rows into a list of (token_tensor, label) pairs without custom classes."""
    dataset: list[tuple[torch.Tensor, int]] = []
    for row in split:
        text = str(row["text"])
        label = int(row["label"])
        ids = encode_text_to_token_ids(text, vocab, max_len)
        dataset.append((torch.tensor(ids, dtype=torch.long), label))
    return dataset


def download_and_split_ag_news_dataset(max_train_samples: int = 12000, max_test_samples: int = 2000, seed: int = 42):
    """Download AG_NEWS dataset from Hugging Face Hub and split into train/valid/test sets."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is missing. Install it via: pip install datasets"
        ) from exc

    print("Downloading/loading AG_NEWS dataset (requires internet on first run)...")
    raw = load_dataset("fancyzhx/ag_news")

    train_source = raw["train"].shuffle(seed=seed)
    test_source = raw["test"].shuffle(seed=seed)

    if max_train_samples and max_train_samples > 0:
        train_source = train_source.select(range(min(max_train_samples, len(train_source))))
    if max_test_samples and max_test_samples > 0:
        test_source = test_source.select(range(min(max_test_samples, len(test_source))))

    split = train_source.train_test_split(test_size=0.05, seed=seed)
    train_split = split["train"]
    valid_split = split["test"]

    return train_split, valid_split, test_source


def create_pytorch_dataloaders(
    train_data: list[tuple[torch.Tensor, int]],
    valid_data: list[tuple[torch.Tensor, int]],
    test_data: list[tuple[torch.Tensor, int]],
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders with custom padding collate function."""
    common_args = {
        "batch_size": batch_size,
        "collate_fn": pad_and_collate_batch,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_data, shuffle=True, **common_args)
    valid_loader = DataLoader(valid_data, shuffle=False, **common_args)
    test_loader = DataLoader(test_data, shuffle=False, **common_args)
    return train_loader, valid_loader, test_loader


# ============================================================================
# 4. Model Layer Composition and Forward Pass Functions
# ============================================================================

def build_transformer_classifier_layers(
    vocab_size: int,
    num_classes: int = 4,
    embedding_dim: int = 100,
    nhead: int = 5,
    num_layers: int = 2,
    dropout: float = 0.1,
    pad_idx: int = 0,
) -> nn.ModuleDict:
    """Build and organize all neural network layers inside an nn.ModuleDict container."""
    if embedding_dim % nhead != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by nhead ({nhead}).")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2 (PAD and UNK tokens).")

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embedding_dim,
        nhead=nhead,
        dim_feedforward=embedding_dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
    )

    layers = nn.ModuleDict({
        "embedding": nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx),
        "dropout": nn.Dropout(dropout),
        "transformer_encoder": nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        ),
        "classifier": nn.Linear(embedding_dim, num_classes),
    })
    return layers


def forward_pass_transformer_classifier(
    layers: nn.ModuleDict,
    pe_table: torch.Tensor,
    token_ids: torch.Tensor,
    pad_idx: int = 0,
    embedding_dim: int = 100,
) -> torch.Tensor:
    """Compute the forward pass through embedding, positional encoding, encoder, and pooling."""
    if token_ids.ndim != 2:
        raise ValueError(
            f"Expected token_ids with shape [batch_size, sequence_length], got {tuple(token_ids.shape)}."
        )

    padding_mask = token_ids.eq(pad_idx)
    x = layers["embedding"](token_ids) * math.sqrt(embedding_dim)

    seq_len = x.size(1)
    if seq_len > pe_table.size(1):
        raise ValueError(
            f"Sequence length ({seq_len}) exceeds positional encoding max_len ({pe_table.size(1)})."
        )

    x = x + pe_table[:, :seq_len, :].to(device=x.device, dtype=x.dtype)
    x = layers["dropout"](x)
    x = layers["transformer_encoder"](x, src_key_padding_mask=padding_mask)

    # Masked Mean Pooling ignoring padding positions
    valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=x.dtype)
    summed = (x * valid_mask).sum(dim=1)
    lengths = valid_mask.sum(dim=1).clamp(min=1.0)
    pooled = summed / lengths

    return layers["classifier"](layers["dropout"](pooled))


# ============================================================================
# 5. Training and Evaluation Pipeline Functions
# ============================================================================

def execute_train_or_eval_epoch(
    layers: nn.ModuleDict,
    pe_table: torch.Tensor,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    pad_idx: int = 0,
    embedding_dim: int = 100,
) -> tuple[float, float]:
    """Execute a single epoch of either training (if optimizer provided) or evaluation."""
    training = optimizer is not None
    layers.train(training)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for token_ids, labels in loader:
            token_ids = token_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = forward_pass_transformer_classifier(
                layers, pe_table, token_ids, pad_idx=pad_idx, embedding_dim=embedding_dim
            )
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                clip_grad_norm_(layers.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += batch_size

    avg_loss = total_loss / max(total_items, 1)
    avg_acc = total_correct / max(total_items, 1)
    return avg_loss, avg_acc


def run_full_training_and_evaluation_pipeline(args) -> None:
    """Execute end-to-end dataset preparation, multi-epoch training, validation, and testing."""
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.max_len < 1:
        raise ValueError("--max-len must be at least 1")

    set_random_seeds_for_reproducibility(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Compute Device]: {device}")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("  Notice: CPU training works, but a CUDA-enabled GPU provides significant acceleration.")

    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download and split data
    train_split, valid_split, test_split = download_and_split_ag_news_dataset(
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    print(f"Sample counts: Train={len(train_split):,}, Valid={len(valid_split):,}, Test={len(test_split):,}")

    # 2. Build vocabulary
    vocab = build_vocabulary_from_texts(
        (str(row["text"]) for row in train_split),
        max_vocab_size=args.max_vocab_size,
    )
    print(f"Vocabulary size: {len(vocab):,}")

    # 3. Create datasets and dataloaders
    train_data = prepare_encoded_news_dataset(train_split, vocab, args.max_len)
    valid_data = prepare_encoded_news_dataset(valid_split, vocab, args.max_len)
    test_data = prepare_encoded_news_dataset(test_split, vocab, args.max_len)
    train_loader, valid_loader, test_loader = create_pytorch_dataloaders(
        train_data, valid_data, test_data, args.batch_size
    )

    # 4. Instantiate model layers and positional encoding table
    layers = build_transformer_classifier_layers(
        vocab_size=len(vocab),
        num_classes=len(LABEL_NAMES),
        embedding_dim=args.embedding_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=PAD_IDX,
    ).to(device)

    pe_table = generate_sinusoidal_positional_encodings(args.embedding_dim, max_len=args.max_len).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        layers.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    history = {
        "train_loss": [], "valid_loss": [],
        "train_accuracy": [], "valid_accuracy": [],
        "learning_rate": [],
    }
    best_valid_accuracy = -1.0
    checkpoint_path = checkpoint_dir / "best_model.pt"
    start_time = time.time()

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = execute_train_or_eval_epoch(
            layers, pe_table, train_loader, criterion, device,
            optimizer=optimizer, pad_idx=PAD_IDX, embedding_dim=args.embedding_dim
        )
        valid_loss, valid_acc = execute_train_or_eval_epoch(
            layers, pe_table, valid_loader, criterion, device,
            optimizer=None, pad_idx=PAD_IDX, embedding_dim=args.embedding_dim
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_accuracy"].append(train_acc)
        history["valid_accuracy"].append(valid_acc)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.2%} | "
            f"valid loss={valid_loss:.4f}, acc={valid_acc:.2%} | "
            f"lr={current_lr:.6f} | {time.time()-epoch_start:.1f}s"
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
                    "max_vocab_size": args.max_vocab_size,
                },
                "best_valid_accuracy": best_valid_accuracy,
            }
            torch.save(checkpoint_payload, checkpoint_path)
            print(f"  --> Saved new best checkpoint: {checkpoint_path}")

    plot_and_save_training_curves(history, results_dir / "training_curves.png")
    with (results_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Final test evaluation with best checkpoint
    print("\nEvaluating best validation checkpoint on the test set...")
    best_ckpt = load_saved_model_checkpoint(checkpoint_path, device)
    clean_state_dict = {
        k: v for k, v in best_ckpt["model_state_dict"].items()
        if not k.startswith("pos_encoder")
    }
    layers.load_state_dict(clean_state_dict, strict=False)

    test_loss, test_acc = execute_train_or_eval_epoch(
        layers, pe_table, test_loader, criterion, device,
        optimizer=None, pad_idx=PAD_IDX, embedding_dim=args.embedding_dim
    )
    elapsed = time.time() - start_time

    print(f"\n================ Final Test Results ================")
    print(f"  Test Loss:          {test_loss:.4f}")
    print(f"  Test Accuracy:      {test_acc:.2%}")
    print(f"  Total Duration:     {elapsed / 60:.2f} minutes")
    print(f"  Checkpoint Path:    {checkpoint_path}")
    print(f"  Training Curves:    {results_dir / 'training_curves.png'}")
    print(f"====================================================\n")

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_validation_accuracy": best_valid_accuracy,
        "training_minutes": elapsed / 60,
        "train_samples": len(train_data),
        "validation_samples": len(valid_data),
        "test_samples": len(test_data),
        "vocab_size": len(vocab),
    }
    with (results_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


# ============================================================================
# 6. Prediction and Inference Functions
# ============================================================================

def predict_news_category_with_probabilities(
    text: str,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> dict:
    """Predict and print class probability distributions for a given news text."""
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
        pad_idx=config["pad_idx"],
    ).to(device)

    clean_state_dict = {
        k: v for k, v in checkpoint["model_state_dict"].items()
        if not k.startswith("pos_encoder")
    }
    layers.load_state_dict(clean_state_dict, strict=False)
    layers.eval()

    pe_table = generate_sinusoidal_positional_encodings(config["embedding_dim"], max_len=config["max_len"]).to(device)

    ids = encode_text_to_token_ids(text, vocab, config["max_len"])
    token_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = forward_pass_transformer_classifier(
            layers, pe_table, token_ids, pad_idx=config["pad_idx"], embedding_dim=config["embedding_dim"]
        )
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_idx = int(probabilities.argmax().item())
    sorted_indices = probabilities.argsort(descending=True).tolist()
    scores = {labels[idx]: float(probabilities[idx].item()) for idx in sorted_indices}

    print("\n--- Prediction Result ---")
    print(f"Text: \"{text}\"")
    print(f"Predicted Category: [{labels[predicted_idx]}]")
    print("Class Probabilities:")
    for category, score in scores.items():
        bar = "=" * int(score * 25)
        print(f"  {category:10s}: {score:6.2%}  [{bar}]")
    print("-------------------------\n")

    return {
        "text": text,
        "predicted_label": labels[predicted_idx],
        "scores": scores,
    }


# ============================================================================
# 7. Architectural Verification (Smoke Test)
# ============================================================================

def run_model_architecture_smoke_test() -> bool:
    """Verify neural network tensor shapes and attention mask path without dataset download."""
    print("\n[Running Functional Smoke Test]...")
    torch.manual_seed(42)
    vocab_size = 500
    batch_size = 4
    sequence_length = 12
    embedding_dim = 100
    max_len = 128

    layers = build_transformer_classifier_layers(
        vocab_size=vocab_size,
        num_classes=4,
        embedding_dim=embedding_dim,
        nhead=5,
        num_layers=2,
        dropout=0.1,
        pad_idx=0,
    )
    layers.eval()

    pe_table = generate_sinusoidal_positional_encodings(embedding_dim=embedding_dim, max_len=max_len)

    fake_token_ids = torch.randint(1, vocab_size, (batch_size, sequence_length))
    fake_token_ids[0, -3:] = 0
    fake_token_ids[1, -5:] = 0

    logits = forward_pass_transformer_classifier(layers, pe_table, fake_token_ids, pad_idx=0, embedding_dim=embedding_dim)

    assert logits.shape == (batch_size, 4), f"Unexpected output shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Model output contains NaN or Inf values"

    print("Smoke test passed successfully.")
    print(f"  - Input shape:  {tuple(fake_token_ids.shape)}  [batch_size, sequence_length]")
    print(f"  - Output shape: {tuple(logits.shape)}  [batch_size, num_classes]")
    print("  - Pure functional Transformer pipeline is verified.\n")
    return True


# ============================================================================
# 8. Command-Line Argument Parser and Main Entry Point
# ============================================================================

def parse_command_line_arguments():
    """Parse user command line arguments."""
    parser = argparse.ArgumentParser(
        description="Transformer News Classifier (Pure Functional Pipeline)"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "test"],
        default="train",
        help="Select mode: 'train', 'predict', or 'test' (default: train).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="English news headline or article text to classify in predict mode.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to saved model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and evaluation.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW optimizer.")
    parser.add_argument("--max-train-samples", type=int, default=12000, help="Max train rows (0 = full split).")
    parser.add_argument("--max-test-samples", type=int, default=2000, help="Max test rows (0 = full split).")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum sequence token length.")
    parser.add_argument("--max-vocab-size", type=int, default=30000, help="Maximum vocabulary capacity.")
    parser.add_argument("--embedding-dim", type=int, default=100, help="Embedding dimension size.")
    parser.add_argument("--nhead", type=int, default=5, help="Number of attention heads.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer Encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR), help="Directory to save checkpoints and results.")

    return parser.parse_args()


def main():
    """Single main entry point for CLI dispatch."""
    args = parse_command_line_arguments()

    if args.text is not None and args.mode == "train":
        args.mode = "predict"

    if args.mode == "test":
        run_model_architecture_smoke_test()

    elif args.mode == "predict":
        text = args.text
        if not text:
            try:
                text = input("Enter a news headline or article in English: ").strip()
            except EOFError:
                text = ""
        if not text:
            print("Please provide text to classify.")
            return
        predict_news_category_with_probabilities(text=text, checkpoint_path=Path(args.checkpoint))

    elif args.mode == "train":
        run_full_training_and_evaluation_pipeline(args)


if __name__ == "__main__":
    main()
