"""Predict an AG_NEWS category for a new news headline/article."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data_utils import PAD_IDX, encode_text
from model import TransformerNewsClassifier

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BASE_DIR / "checkpoints" / "best_model.pt"


def load_checkpoint(path: Path, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\nTrain the model first with: python train.py"
        )
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Classify a news text with the trained Transformer.")
    parser.add_argument("--text", type=str, help="News headline or article text to classify.")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    text = args.text or input("Enter a news headline/article in English: ").strip()
    if not text:
        raise SystemExit("Please provide some text to classify.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(Path(args.checkpoint), device)
    vocab = checkpoint["vocab"]
    labels = checkpoint["labels"]
    config = checkpoint["config"]

    model = TransformerNewsClassifier(
        vocab_size=config["vocab_size"],
        num_classes=config["num_classes"],
        embedding_dim=config["embedding_dim"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        pad_idx=config["pad_idx"],
        max_len=config["max_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ids = encode_text(text, vocab, config["max_len"])
    token_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(token_ids)
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_idx = int(probabilities.argmax().item())
    print("\n--- Prediction ---")
    print(f"Text: {text}")
    print(f"Predicted category: {labels[predicted_idx]}")
    print("Scores:")
    for idx in probabilities.argsort(descending=True).tolist():
        print(f"  {labels[idx]:8s}: {probabilities[idx].item():.2%}")
    print("\nNote: Scores are model estimates, not guaranteed probabilities of factual correctness.")


if __name__ == "__main__":
    main()
