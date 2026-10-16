"""Train and evaluate a Transformer Encoder on AG_NEWS."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from data_utils import (
    LABEL_NAMES,
    PAD_IDX,
    NewsDataset,
    build_vocab,
    load_ag_news,
    make_dataloaders,
)
from model import TransformerNewsClassifier

BASE_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
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

            logits = model(token_ids)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += batch_size

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def save_curves(history: dict[str, list[float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["valid_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history["valid_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer Encoder news classifier.")
    parser.add_argument("--max-train-samples", type=int, default=12000,
                        help="Number of rows sampled from AG_NEWS train before validation split; 0=all.")
    parser.add_argument("--max-test-samples", type=int, default=2000,
                        help="Number of test rows; 0=all.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--nhead", type=int, default=5)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(BASE_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.max_len < 1:
        raise ValueError("--max-len must be at least 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Tip: CPU training works, but a CUDA-enabled PyTorch installation can be faster on an NVIDIA GPU.")

    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_split, valid_split, test_split = load_ag_news(
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    print(f"Rows: train={len(train_split)}, validation={len(valid_split)}, test={len(test_split)}")

    vocab = build_vocab(
        (str(row["text"]) for row in train_split),
        max_vocab_size=args.max_vocab_size,
    )
    print(f"Vocabulary size: {len(vocab):,}")

    train_dataset = NewsDataset(train_split, vocab, args.max_len)
    valid_dataset = NewsDataset(valid_split, vocab, args.max_len)
    test_dataset = NewsDataset(test_split, vocab, args.max_len)
    train_loader, valid_loader, test_loader = make_dataloaders(
        train_dataset, valid_dataset, test_dataset, args.batch_size
    )

    model = TransformerNewsClassifier(
        vocab_size=len(vocab),
        num_classes=len(LABEL_NAMES),
        embedding_dim=args.embedding_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=PAD_IDX,
        max_len=args.max_len,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # AdamW with a modest learning rate is a stable default for this mini-project.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
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

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_loss, valid_acc = run_epoch(model, valid_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_accuracy"].append(train_acc)
        history["valid_accuracy"].append(valid_acc)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss={train_loss:.4f}, train acc={train_acc:.3%} | "
            f"valid loss={valid_loss:.4f}, valid acc={valid_acc:.3%} | "
            f"lr={current_lr:.6f} | {time.time()-epoch_start:.1f}s"
        )

        if valid_acc > best_valid_accuracy:
            best_valid_accuracy = valid_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
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
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved best model: {checkpoint_path}")

    # Save curves and metrics.
    save_curves(history, results_dir / "training_curves.png")
    with (results_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Evaluate the checkpoint with the best validation accuracy, not merely the last epoch.
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    elapsed = time.time() - start_time
    print("\nFinal test results (best validation checkpoint):")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.3%}")
    print(f"Total training time: {elapsed/60:.1f} minutes")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Curves: {results_dir / 'training_curves.png'}")

    metrics = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "best_validation_accuracy": best_valid_accuracy,
        "training_minutes": elapsed / 60,
        "train_samples": len(train_dataset),
        "validation_samples": len(valid_dataset),
        "test_samples": len(test_dataset),
        "vocab_size": len(vocab),
    }
    with (results_dir / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
