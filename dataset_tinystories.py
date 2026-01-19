"""
TinyStories Dataset for training recursive transformer on language modeling.

Streams from HuggingFace: "roneneldan/TinyStories"
Standard autoregressive setup: predict next token at each position.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from typing import Iterator, Dict, Optional
from datasets import load_dataset


def get_tokenizer(model_name: str = "gpt2"):
    """Load the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


class TinyStoriesDataset(IterableDataset):
    """
    TinyStories dataset for autoregressive language modeling.

    Streams from HuggingFace to avoid downloading the full dataset.
    Standard next-token prediction: loss computed on ALL tokens.

    Each sample:
        Input:  [t0, t1, t2, ..., tn-1]
        Target: [t1, t2, t3, ..., tn]
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 256,
        split: str = "train",
        buffer_size: int = 10000
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_seq_len: Maximum sequence length (TinyStories has short stories)
            split: "train" or "validation"
            buffer_size: Shuffle buffer size for streaming
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split
        self.buffer_size = buffer_size

        # Load streaming dataset
        self.dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            streaming=True
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield tokenized samples."""
        # Shuffle for training
        if self.split == "train":
            dataset = self.dataset.shuffle(buffer_size=self.buffer_size)
        else:
            dataset = self.dataset

        for sample in dataset:
            text = sample['text']

            # Tokenize
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_len + 1,  # +1 because we'll shift
                truncation=True
            )

            # Skip very short sequences
            if len(tokens) < 10:
                continue

            # Pad if needed
            if len(tokens) < self.max_seq_len + 1:
                pad_len = self.max_seq_len + 1 - len(tokens)
                tokens = tokens + [self.tokenizer.pad_token_id] * pad_len

            # Convert to tensors
            tokens = torch.tensor(tokens[:self.max_seq_len + 1], dtype=torch.long)

            # Input: all but last token
            # Target: all but first token (shifted by 1)
            input_ids = tokens[:-1]  # [max_seq_len]
            target_ids = tokens[1:]  # [max_seq_len]

            # Attention mask (1 for real tokens, 0 for padding)
            # Note: we look at input_ids to determine padding
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            # Don't compute loss on padding tokens
            # Set target to -100 where input was padding
            target_ids = target_ids.clone()
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100

            yield {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'attention_mask': attention_mask,
            }


class TinyStoriesDatasetFinite(torch.utils.data.Dataset):
    """
    Finite version of TinyStories for validation.
    Downloads and caches a subset of the dataset.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 256,
        split: str = "validation",
        max_samples: int = 1000
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split

        # Load and cache samples
        print(f"Loading {max_samples} samples from TinyStories {split}...")
        dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            streaming=True
        )

        self.samples = []
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            text = sample['text']
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_len + 1,
                truncation=True
            )
            if len(tokens) >= 10:
                self.samples.append(tokens)

        print(f"Loaded {len(self.samples)} valid samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.samples[idx]

        # Pad if needed
        if len(tokens) < self.max_seq_len + 1:
            pad_len = self.max_seq_len + 1 - len(tokens)
            tokens = tokens + [self.tokenizer.pad_token_id] * pad_len

        tokens = torch.tensor(tokens[:self.max_seq_len + 1], dtype=torch.long)

        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        target_ids = target_ids.clone()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
        }


def create_tinystories_dataloaders(
    tokenizer_name: str = "gpt2",
    max_seq_len: int = 256,
    batch_size: int = 64,
    num_workers: int = 0,  # Must be 0 for IterableDataset
    val_samples: int = 1000
):
    """
    Create train and validation dataloaders for TinyStories.

    Args:
        tokenizer_name: HuggingFace tokenizer name
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of workers (must be 0 for streaming)
        val_samples: Number of validation samples to cache

    Returns:
        train_loader, val_loader, tokenizer
    """
    tokenizer = get_tokenizer(tokenizer_name)

    # Training: streaming dataset
    train_dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split="train"
    )

    # Validation: finite cached dataset
    val_dataset = TinyStoriesDatasetFinite(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split="validation",
        max_samples=val_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, tokenizer


def get_token_frequencies(tokenizer, num_samples: int = 10000) -> Dict[int, int]:
    """
    Compute token frequency counts from TinyStories.

    Used for analyzing whether rare tokens use more iterations.
    """
    from collections import Counter

    print(f"Computing token frequencies from {num_samples} samples...")

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True
    )

    token_counts = Counter()

    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        tokens = tokenizer.encode(sample['text'], add_special_tokens=False)
        token_counts.update(tokens)

    return dict(token_counts)


if __name__ == '__main__':
    print("Testing TinyStories dataset...")

    tokenizer = get_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Test streaming dataset
    print("\nTesting streaming dataset...")
    train_loader, val_loader, tok = create_tinystories_dataloaders(
        batch_size=4,
        max_seq_len=128
    )

    # Get one batch
    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch target_ids shape: {batch['target_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")

    # Decode first sample
    decoded = tok.decode(batch['input_ids'][0], skip_special_tokens=True)
    print(f"\nSample story: {decoded[:200]}...")

    # Test validation
    print(f"\nValidation set size: {len(val_loader.dataset)}")

    print("\nDone!")
