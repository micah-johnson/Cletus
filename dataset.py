"""
Dataset for training recursive transformer on language tasks.

Supports:
- GSM8K math word problems (loaded from jsonl files)
- Llama tokenizer (32k vocab)
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def get_tokenizer(model_name: str = "gpt2"):
    """
    Load the tokenizer.

    Default: GPT-2 tokenizer (50257 vocab, no auth required)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure pad token exists (GPT-2 doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


class GSM8KDataset(Dataset):
    """
    GSM8K Math Word Problem Dataset.

    Task: Given a math question, predict the answer.
    Format: "Question: {question}\nAnswer: {answer}"
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_len: int = 512,
        split: str = "train"
    ):
        """
        Args:
            data_path: Path to the jsonl file
            tokenizer: HuggingFace tokenizer
            max_seq_len: Maximum sequence length
            split: "train" or "test" (for logging)
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split

        # Load data from jsonl
        self.data = self._load_jsonl(data_path)
        print(f"Loaded {len(self.data)} samples from {split} split")

    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load data from jsonl file."""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
                - input_ids: [seq_len] tokenized input (question + prompt for answer)
                - target_ids: [seq_len] tokenized target (full sequence with answer)
                - attention_mask: [seq_len] attention mask
        """
        sample = self.data[idx]
        question = sample['question']
        answer = sample['answer']

        # Format: we train the model to complete the answer
        # Input has the question, target has question + answer
        input_text = f"Question: {question}\nAnswer:"
        target_text = f"Question: {question}\nAnswer: {answer}"

        # Tokenize input (what the model sees)
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize target (what we want the model to predict)
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # For language modeling, we typically use the same sequence for input and target
        # but shift by one position. Here we'll use the full target for supervision.
        input_ids = target_encoding['input_ids'].squeeze(0)
        attention_mask = target_encoding['attention_mask'].squeeze(0)

        # Target is the same as input but shifted (handled in loss computation)
        # We mask out the question part so we only compute loss on the answer
        target_ids = input_ids.clone()

        # Find where the answer starts (after "Answer:")
        input_len = len(self.tokenizer.encode(input_text, add_special_tokens=False))

        # Set target to -100 (ignore) for the prompt portion
        # This way we only train on predicting the answer
        target_ids[:min(input_len, self.max_seq_len)] = -100

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'question': question,
            'answer': answer
        }


class GSM8KDatasetSimple(Dataset):
    """
    Autoregressive dataset for next-token prediction.

    Training setup:
        Input:  [Question tokens] [Answer tokens]
        Target: [IGNORE...] [Answer tokens shifted by 1] [EOS]

    The model learns to predict the next token at each position.
    Loss is only computed on answer tokens (question is context).

    Example:
        Question: "What is (5 + 3)?"
        Answer: "The answer is 8"

        Input:  [What, is, (, 5, +, 3, ), ?, The, answer, is, 8]
        Target: [IGN,  IGN, IGN, IGN, IGN, IGN, IGN, The, answer, is, 8, EOS]

        Position 7 (after "?") learns to predict "The"
        Position 11 (after "is") learns to predict "8" <- needs more iterations!
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_input_len: int = 256,
        max_output_len: int = 32,
        split: str = "train"
    ):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.max_seq_len = max_input_len + max_output_len
        self.split = split

        self.data = self._load_jsonl(data_path)
        print(f"Loaded {len(self.data)} samples from {split} split")

    def _load_jsonl(self, path: str) -> List[Dict]:
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        question = sample['question']
        answer = str(sample['answer'])

        # Encode question
        question_tokens = self.tokenizer.encode(
            question,
            add_special_tokens=True,
            max_length=self.max_input_len,
            truncation=True
        )

        # Encode answer
        answer_tokens = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
            max_length=self.max_output_len - 1,  # Leave room for EOS
            truncation=True
        )

        # Full sequence: question + answer + EOS
        eos_id = self.tokenizer.eos_token_id
        full_sequence = question_tokens + answer_tokens + [eos_id]

        # For autoregressive: target is input shifted by 1
        # Input:  [q0, q1, ..., qn, a0, a1, ..., am, EOS]
        # Target: [q1, q2, ..., qn, a0, a1, ..., am, EOS, PAD] but we mask question
        #
        # Actually simpler: target[i] = input[i+1], and we mask question positions

        seq_len = len(full_sequence)
        question_len = len(question_tokens)
        answer_len = len(answer_tokens) + 1  # +1 for EOS

        # Pad to max length
        pad_len = self.max_seq_len - seq_len
        if pad_len > 0:
            input_sequence = full_sequence + [self.tokenizer.pad_token_id] * pad_len
        else:
            input_sequence = full_sequence[:self.max_seq_len]
            seq_len = self.max_seq_len

        input_ids = torch.tensor(input_sequence, dtype=torch.long)

        # Target: shifted by 1 (next token prediction)
        # target[i] = what should be predicted at position i = input[i+1]
        target_ids = torch.full((self.max_seq_len,), -100, dtype=torch.long)

        # Fill in targets for answer positions only
        # Position (question_len - 1) should predict first answer token
        # Position (question_len) should predict second answer token, etc.
        for i in range(answer_len):
            src_pos = question_len + i  # Position in input that has the answer token
            tgt_pos = question_len - 1 + i  # Position that should predict this token

            if tgt_pos < self.max_seq_len and src_pos < len(full_sequence):
                target_ids[tgt_pos] = full_sequence[src_pos]

        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:min(seq_len, self.max_seq_len)] = 1

        # Get depth if available
        depth = sample.get('depth', 0)

        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask,
            'question': question,
            'answer': answer,
            'answer_start': question_len - 1,  # First position that predicts answer
            'answer_len': answer_len,
            'depth': depth
        }


def create_gsm8k_dataloaders(
    data_dir: str = "data/gsm8k",
    tokenizer_name: str = "gpt2",
    max_seq_len: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
    simple_mode: bool = True
) -> Tuple[DataLoader, DataLoader, 'AutoTokenizer']:
    """
    Create train and test dataloaders for GSM8K.

    Args:
        data_dir: Directory containing train.jsonl and test.jsonl
        tokenizer_name: HuggingFace tokenizer name
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of dataloader workers
        simple_mode: If True, use GSM8KDatasetSimple

    Returns:
        train_loader, test_loader, tokenizer
    """
    tokenizer = get_tokenizer(tokenizer_name)

    train_path = os.path.join(data_dir, "train.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    DatasetClass = GSM8KDatasetSimple if simple_mode else GSM8KDataset

    if simple_mode:
        train_dataset = DatasetClass(
            train_path, tokenizer,
            max_input_len=max_seq_len - 32,
            max_output_len=32,
            split="train"
        )
        test_dataset = DatasetClass(
            test_path, tokenizer,
            max_input_len=max_seq_len - 32,
            max_output_len=32,
            split="test"
        )
    else:
        train_dataset = DatasetClass(train_path, tokenizer, max_seq_len, split="train")
        test_dataset = DatasetClass(test_path, tokenizer, max_seq_len, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, tokenizer


# =============================================================================
# Legacy: Arithmetic Dataset (for backwards compatibility)
# =============================================================================

class ArithmeticTokenizer:
    """Simple character-level tokenizer for arithmetic expressions."""

    def __init__(self):
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.digits = list('0123456789')
        self.operators = ['+', '-', '*']
        self.brackets = ['(', ')']
        self.other = [' ', '=']

        self.vocab = (
            self.special_tokens +
            self.digits +
            self.operators +
            self.brackets +
            self.other
        )

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

        self.pad_token_id = self.token_to_id['<pad>']
        self.sos_token_id = self.token_to_id['<sos>']
        self.eos_token_id = self.token_to_id['<eos>']
        self.unk_token_id = self.token_to_id['<unk>']

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = []
        if add_special_tokens:
            ids.append(self.sos_token_id)
        for char in text:
            ids.append(self.token_to_id.get(char, self.unk_token_id))
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for id in ids:
            token = self.id_to_token.get(id, '<unk>')
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return ''.join(tokens)

    def pad_sequence(self, ids: List[int], max_len: int) -> List[int]:
        if len(ids) > max_len:
            return ids[:max_len]
        return ids + [self.pad_token_id] * (max_len - len(ids))


if __name__ == '__main__':
    print("Testing GSM8K dataset...")

    # Test tokenizer loading
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # Test dataset
    print("\nTesting GSM8K dataset...")
    try:
        train_loader, test_loader, tok = create_gsm8k_dataloaders(
            batch_size=2,
            num_workers=0,
            simple_mode=True
        )

        batch = next(iter(train_loader))
        print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"Batch target_ids shape: {batch['target_ids'].shape}")
        print(f"\nSample question: {batch['question'][0][:100]}...")
        print(f"Sample answer: {batch['answer'][0]}")

        # Decode to verify
        decoded = tok.decode(batch['input_ids'][0], skip_special_tokens=True)
        print(f"\nDecoded input: {decoded[:200]}...")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure data/gsm8k/train.jsonl and test.jsonl exist")
