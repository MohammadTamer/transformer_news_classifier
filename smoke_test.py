"""Quick test of tensor dimensions without downloading AG_NEWS."""

import torch
from model import TransformerNewsClassifier


def main():
    torch.manual_seed(42)
    vocab_size = 500
    batch_size = 4
    sequence_length = 12
    model = TransformerNewsClassifier(
        vocab_size=vocab_size,
        num_classes=4,
        embedding_dim=100,
        nhead=5,
        num_layers=2,
        dropout=0.1,
        pad_idx=0,
        max_len=128,
    )
    fake_token_ids = torch.randint(1, vocab_size, (batch_size, sequence_length))
    # Add a few padding positions to make sure the padding mask path works.
    fake_token_ids[0, -3:] = 0
    fake_token_ids[1, -5:] = 0
    logits = model(fake_token_ids)
    assert logits.shape == (batch_size, 4), f"Unexpected output shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "Model output contains NaN/Inf values"
    print("Smoke test passed.")
    print(f"Input shape:  {tuple(fake_token_ids.shape)}  (batch, sequence)")
    print(f"Output shape: {tuple(logits.shape)}  (batch, classes)")
    print("The model dimensions and padding-mask path are working.")


if __name__ == "__main__":
    main()
