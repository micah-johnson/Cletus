"""
OpenWebText dataset for training, FineWeb for validation.

OpenWebText: ~8M documents for exactly 1 epoch of training
FineWeb: Small validation set from a different distribution
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple


# Use GPT-2 tokenizer (50257 vocab)
GPT2_VOCAB_SIZE = 50257


def get_tokenizer(name: str = "gpt2"):
    """Load tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class OpenWebTextDataset(IterableDataset):
    """
    Streaming dataset for OpenWebText - exactly 1 epoch.

    Uses HuggingFace datasets streaming to avoid loading entire dataset into memory.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 512,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split

        # Load streaming dataset
        self.dataset = load_dataset(
            "Skylion007/openwebtext",
            split=split,
            streaming=True,
            trust_remote_code=True
        )

    def __iter__(self):
        buffer = []

        for example in self.dataset:
            text = example['text']

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(tokens)

            # Yield chunks of max_seq_len + 1 (for input/target offset)
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[:self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)

                yield {
                    'input_ids': input_ids,
                    'target_ids': target_ids,
                }


class FineWebValidationDataset(Dataset):
    """
    Finite validation dataset from FineWeb.

    Caches a fixed number of samples for consistent validation.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 512,
        max_samples: int = 1000,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        print(f"Loading FineWeb validation set ({max_samples} samples)...")

        # Load streaming and cache samples
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",  # Use the 10B token sample
            split="train",
            streaming=True,
            trust_remote_code=True
        )

        buffer = []
        samples_collected = 0

        for example in dataset:
            if samples_collected >= max_samples:
                break

            text = example['text']
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            buffer.extend(tokens)

            while len(buffer) >= self.max_seq_len + 1 and samples_collected < max_samples:
                chunk = buffer[:self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:], dtype=torch.long)

                self.samples.append({
                    'input_ids': input_ids,
                    'target_ids': target_ids,
                })
                samples_collected += 1

        print(f"Loaded {len(self.samples)} validation samples from FineWeb")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_openwebtext_dataloaders(
    tokenizer_name: str = "gpt2",
    max_seq_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 4,
    val_samples: int = 1000,
) -> Tuple[DataLoader, DataLoader, any]:
    """
    Create train and validation dataloaders.

    Args:
        tokenizer_name: HuggingFace tokenizer name
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        val_samples: Number of validation samples

    Returns:
        train_loader, val_loader, tokenizer
    """
    tokenizer = get_tokenizer(tokenizer_name)

    # Training: OpenWebText streaming (1 epoch)
    train_dataset = OpenWebTextDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        split="train",
    )

    # Validation: FineWeb cached samples
    val_dataset = FineWebValidationDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        max_samples=val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, tokenizer


def estimate_openwebtext_tokens():
    """Estimate total tokens in OpenWebText for progress tracking."""
    # OpenWebText has ~8M documents, average ~500 tokens each
    # Total: ~4B tokens
    # With seq_len=512, that's ~8M training samples
    return 4_000_000_000  # 4B tokens estimate


if __name__ == '__main__':
    print("Testing OpenWebText dataset...")

    tokenizer = get_tokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Test train dataset
    train_dataset = OpenWebTextDataset(tokenizer, max_seq_len=512)

    print("\nSampling from OpenWebText:")
    count = 0
    for sample in train_dataset:
        print(f"  Input shape: {sample['input_ids'].shape}")
        print(f"  Sample text: {tokenizer.decode(sample['input_ids'][:50])}...")
        count += 1
        if count >= 3:
            break

    # Test val dataset
    print("\nLoading FineWeb validation set:")
    val_dataset = FineWebValidationDataset(tokenizer, max_seq_len=512, max_samples=10)
    print(f"  Loaded {len(val_dataset)} samples")

    print("\nDone!")
