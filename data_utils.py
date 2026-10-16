"""Dataset download, tokenization, vocabulary, and DataLoader utilities."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
PAD_IDX = 0
UNK_IDX = 1

_TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[^\w\s]", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """A small, dependency-free English tokenizer."""
    return _TOKEN_PATTERN.findall(text.lower())


def load_ag_news(max_train_samples: int = 12000, max_test_samples: int = 2000, seed: int = 42):
    """Download AG_NEWS from the Hugging Face Hub.

    max_train_samples=0 or max_test_samples=0 means use the full split.
    A seeded shuffle happens before selecting a smaller subset, to avoid
    accidentally taking only one class if the source rows are grouped by label.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is missing. Run: pip install -r requirements.txt"
        ) from exc

    print("Downloading/loading AG_NEWS dataset (the first run needs internet)...")
    raw = load_dataset("fancyzhx/ag_news")
    train_source = raw["train"].shuffle(seed=seed)
    test_source = raw["test"].shuffle(seed=seed)

    if max_train_samples and max_train_samples > 0:
        train_source = train_source.select(range(min(max_train_samples, len(train_source))))
    if max_test_samples and max_test_samples > 0:
        test_source = test_source.select(range(min(max_test_samples, len(test_source))))

    # Keep validation separate from the training data.
    split = train_source.train_test_split(test_size=0.05, seed=seed)
    train_split = split["train"]
    valid_split = split["test"]
    return train_split, valid_split, test_source


def build_vocab(
    texts: Iterable[str],
    max_vocab_size: int = 30000,
    min_freq: int = 1,
) -> dict[str, int]:
    """Build a vocabulary from TRAINING text only."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    available = max(0, max_vocab_size - len(vocab))
    for token, count in counter.most_common():
        if count < min_freq or len(vocab) - 2 >= available:
            break
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    tokens = tokenize(text)[:max_len]
    ids = [vocab.get(token, UNK_IDX) for token in tokens]
    # Ensure every sample contains at least one token.
    return ids if ids else [UNK_IDX]


class NewsDataset(Dataset):
    """Converts a Hugging Face split into token-ID sequences and integer labels."""

    def __init__(self, split, vocab: dict[str, int], max_len: int = 128):
        self.examples: list[tuple[list[int], int]] = []
        self.vocab = vocab
        self.max_len = max_len
        for row in split:
            text = str(row["text"])
            label = int(row["label"])  # AG_NEWS labels are 0, 1, 2, 3.
            self.examples.append((encode_text(text, vocab, max_len), label))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        ids, label = self.examples[idx]
        return torch.tensor(ids, dtype=torch.long), label


def collate_batch(batch: Sequence[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad each batch to its longest sequence. Returns (token_ids, labels)."""
    sequences, labels = zip(*batch)
    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, label_tensor


def make_dataloaders(
    train_dataset: Dataset,
    valid_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
):
    common = {
        "batch_size": batch_size,
        "collate_fn": collate_batch,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    valid_loader = DataLoader(valid_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, valid_loader, test_loader
