"""
Wikipedia Dataset for training recursive transformer on language modeling.

Streams from HuggingFace: "omarkamali/wikipedia-monthly"
Standard autoregressive setup: predict next token at each position.

Uses GPT-2 tokenizer with full vocab (50257 tokens).
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from typing import Iterator, Dict, Optional
from datasets import load_dataset


# GPT-2 vocab size
GPT2_VOCAB_SIZE = 50257


def get_tokenizer(model_name: str = "gpt2"):
    """Load the GPT-2 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


class WikipediaDataset(IterableDataset):
    """
    Wikipedia dataset for autoregressive language modeling.

    Streams from HuggingFace to avoid downloading the full dataset.
    Standard next-token prediction: loss computed on ALL tokens.

    Each sample:
        Input:  [t0, t1, t2, ..., tn-1]
        Target: [t1, t2, t3, ..., tn]
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 512,
        dataset_name: str = "omarkamali/wikipedia-monthly",
        dataset_config: str = "20251201.en",
        split: str = "train",
        buffer_size: int = 10000,
        text_column: str = "text",
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_seq_len: Maximum sequence length
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration (e.g., "20251201.en")
            split: Dataset split
            buffer_size: Shuffle buffer size for streaming
            text_column: Name of the text column in the dataset
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.split = split
        self.text_column = text_column

        # Load streaming dataset
        print(f"Loading Wikipedia dataset: {dataset_name}/{dataset_config}")
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True
        )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Yield tokenized samples."""
        # Shuffle for training
        dataset = self.dataset.shuffle(buffer_size=self.buffer_size)

        for sample in dataset:
            text = sample[self.text_column]

            # Skip empty texts
            if not text or len(text.strip()) < 50:
                continue

            # Tokenize
            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_len + 1,  # +1 because we'll shift
                truncation=True
            )

            # Skip very short sequences
            if len(tokens) < 32:
                continue

            # Pad if needed
            pad_id = self.tokenizer.pad_token_id
            if len(tokens) < self.max_seq_len + 1:
                pad_len = self.max_seq_len + 1 - len(tokens)
                tokens = tokens + [pad_id] * pad_len

            # Convert to tensors
            tokens = torch.tensor(tokens[:self.max_seq_len + 1], dtype=torch.long)

            # Input: all but last token
            # Target: all but first token (shifted by 1)
            input_ids = tokens[:-1]  # [max_seq_len]
            target_ids = tokens[1:]  # [max_seq_len]

            # Attention mask (1 for real tokens, 0 for padding)
            attention_mask = (input_ids != pad_id).long()

            # Don't compute loss on padding tokens
            target_ids = target_ids.clone()
            target_ids[target_ids == pad_id] = -100

            yield {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'attention_mask': attention_mask,
            }


class WikipediaDatasetFinite(torch.utils.data.Dataset):
    """
    Finite version of Wikipedia for validation.
    Downloads and caches a subset of the dataset.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 512,
        dataset_name: str = "omarkamali/wikipedia-monthly",
        dataset_config: str = "20251201.en",
        split: str = "train",  # Wikipedia monthly doesn't have validation split
        max_samples: int = 1000,
        text_column: str = "text",
        skip_samples: int = 100000,  # Skip first N samples to get different data for val
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Load and cache samples
        print(f"Loading {max_samples} validation samples from Wikipedia (skipping first {skip_samples})...")
        dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=True
        )

        self.samples = []
        for i, sample in enumerate(dataset):
            # Skip first N samples to use different data for validation
            if i < skip_samples:
                continue
            if i >= skip_samples + max_samples * 2:  # Load extra in case some are too short
                break

            text = sample[text_column]
            if not text or len(text.strip()) < 50:
                continue

            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=True,
                max_length=self.max_seq_len + 1,
                truncation=True
            )
            if len(tokens) >= 32:
                self.samples.append(tokens)

            if len(self.samples) >= max_samples:
                break

        print(f"Loaded {len(self.samples)} valid samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.samples[idx].copy()

        # Pad if needed
        pad_id = self.tokenizer.pad_token_id
        if len(tokens) < self.max_seq_len + 1:
            pad_len = self.max_seq_len + 1 - len(tokens)
            tokens = tokens + [pad_id] * pad_len

        tokens = torch.tensor(tokens[:self.max_seq_len + 1], dtype=torch.long)

        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        attention_mask = (input_ids != pad_id).long()

        target_ids = target_ids.clone()
        target_ids[target_ids == pad_id] = -100

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
        }


def create_wikipedia_dataloaders(
    tokenizer_name: str = "gpt2",
    max_seq_len: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,  # Must be 0 for IterableDataset
    val_samples: int = 1000,
    dataset_name: str = "omarkamali/wikipedia-monthly",
    dataset_config: str = "20251201.en",
):
    """
    Create train and validation dataloaders for Wikipedia.

    Args:
        tokenizer_name: HuggingFace tokenizer name
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of workers (must be 0 for streaming)
        val_samples: Number of validation samples to cache
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration

    Returns:
        train_loader, val_loader, tokenizer
    """
    tokenizer = get_tokenizer(tokenizer_name)

    # Training: streaming dataset
    train_dataset = WikipediaDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split="train",
    )

    # Validation: finite cached dataset (from later in the stream)
    val_dataset = WikipediaDatasetFinite(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split="train",  # Use train split but skip first N samples
        max_samples=val_samples,
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


if __name__ == '__main__':
    print("Testing Wikipedia dataset...")

    tokenizer = get_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Test streaming dataset
    print("\nTesting streaming dataset...")
    train_loader, val_loader, tok = create_wikipedia_dataloaders(
        batch_size=4,
        max_seq_len=256,
        val_samples=100
    )

    # Get one batch
    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch target_ids shape: {batch['target_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")

    # Decode first sample
    decoded = tok.decode(batch['input_ids'][0], skip_special_tokens=True)
    print(f"\nSample text: {decoded[:300]}...")

    # Test validation
    print(f"\nValidation set size: {len(val_loader.dataset)}")

    print("\nDone!")
