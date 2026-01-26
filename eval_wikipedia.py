#!/usr/bin/env python3
"""
Evaluate Wikipedia model on validation dataset.

Automatically loads the best Wikipedia model checkpoint and evaluates
on the wikipedia-monthly validation set (or optionally wikitext-103).

Usage:
    python eval_wikipedia.py                              # Auto-load best model
    python eval_wikipedia.py --checkpoint path/to/model.pt
    python eval_wikipedia.py --val-samples 5000           # More validation samples
    python eval_wikipedia.py --dataset wikitext-103       # Use WikiText-103 instead
"""

import os
import glob
import argparse
from typing import Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FlashRecursiveTransformer
from dataset_wikipedia import (
    WikipediaDatasetFinite,
    get_tokenizer,
    GPT2_VOCAB_SIZE,
)


# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = 'checkpoints_wikipedia_125m'

# Default model config (fallback if not in checkpoint)
MODEL_125M_CONFIG = {
    'vocab_size': GPT2_VOCAB_SIZE,
    'd_model': 704,
    'n_heads': 8,
    'n_layers': 9,
    'd_ff': 2816,
    'max_iterations': 5,
    'dropout': 0.1,
    'max_seq_len': 512,
}


def find_best_checkpoint(checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> Optional[str]:
    """
    Find the best model checkpoint in the given directory.

    Priority:
    1. best_model.pt (saved during training when val loss improves)
    2. final_model.pt (saved at end of training)
    3. Latest checkpoint_*.pt by step number

    Returns:
        Path to the best checkpoint, or None if not found.
    """
    # Check for best_model.pt first
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_path):
        return best_path

    # Check for final_model.pt
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    if os.path.exists(final_path):
        return final_path

    # Find latest checkpoint by step number
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.pt'))
    if checkpoint_files:
        # Extract step numbers and sort
        def get_step(path):
            basename = os.path.basename(path)
            step_str = basename.replace('checkpoint_', '').replace('.pt', '')
            try:
                return int(step_str)
            except ValueError:
                return -1

        checkpoint_files.sort(key=get_step, reverse=True)
        return checkpoint_files[0]

    return None


def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[FlashRecursiveTransformer, Dict]:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config_dict)
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use defaults
    config = checkpoint.get('config', MODEL_125M_CONFIG)

    # Handle both dict and object configs
    if hasattr(config, '__dict__'):
        config = vars(config)

    print(f"Model config: d_model={config['d_model']}, n_layers={config['n_layers']}, "
          f"n_heads={config['n_heads']}, max_iterations={config['max_iterations']}")

    # Create model
    model = FlashRecursiveTransformer(
        vocab_size=config.get('vocab_size', GPT2_VOCAB_SIZE),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config.get('d_ff', config['d_model'] * 4),
        max_iterations=config['max_iterations'],
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 512),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Print checkpoint info
    if 'best_val_loss' in checkpoint:
        print(f"Checkpoint best val loss: {checkpoint['best_val_loss']:.4f}")
    if 'global_step' in checkpoint:
        print(f"Checkpoint training step: {checkpoint['global_step']}")

    return model, config


def create_wikitext103_dataloader(
    tokenizer,
    max_seq_len: int = 512,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create WikiText-103 validation dataloader.

    Uses the standard WikiText-103 validation split from HuggingFace.
    """
    from datasets import load_dataset

    print("Loading WikiText-103 validation set...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")

    # Process samples
    samples = []
    for item in dataset:
        text = item['text']
        if not text or len(text.strip()) < 50:
            continue

        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_seq_len + 1,
            truncation=True
        )

        if len(tokens) >= 32:
            samples.append(tokens)

    print(f"Loaded {len(samples)} valid samples from WikiText-103 validation")

    # Create simple dataset
    class WikiText103Dataset(torch.utils.data.Dataset):
        def __init__(self, samples, tokenizer, max_seq_len):
            self.samples = samples
            self.tokenizer = tokenizer
            self.max_seq_len = max_seq_len
            self.pad_id = tokenizer.pad_token_id

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            tokens = self.samples[idx].copy() if isinstance(self.samples[idx], list) else list(self.samples[idx])

            # Pad if needed
            if len(tokens) < self.max_seq_len + 1:
                pad_len = self.max_seq_len + 1 - len(tokens)
                tokens = tokens + [self.pad_id] * pad_len

            tokens = torch.tensor(tokens[:self.max_seq_len + 1], dtype=torch.long)

            input_ids = tokens[:-1]
            target_ids = tokens[1:]

            attention_mask = (input_ids != self.pad_id).long()

            target_ids = target_ids.clone()
            target_ids[target_ids == self.pad_id] = -100

            return {
                'input_ids': input_ids,
                'target_ids': target_ids,
                'attention_mask': attention_mask,
            }

    val_dataset = WikiText103Dataset(samples, tokenizer, max_seq_len)

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def evaluate(
    model: FlashRecursiveTransformer,
    val_loader: DataLoader,
    device: str = 'cuda',
    use_amp: bool = True,
) -> Dict:
    """
    Evaluate model on validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation dataloader
        device: Device to run on
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_iterations = 0.0
    iteration_counts = defaultdict(int)
    num_batches = 0

    # Per-iteration loss tracking
    per_iter_losses = defaultdict(float)
    per_iter_counts = defaultdict(int)

    pbar = tqdm(val_loader, desc="Evaluating", unit="batch")

    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        batch_size, seq_len = input_ids.shape

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                output, metadata = model(
                    input_ids,
                    attention_mask=attention_mask,
                    force_iterations=5,  # Force 5 iterations
                )

                # Compute loss
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)),
                    target_ids.view(-1),
                    ignore_index=-100,
                    reduction='sum'
                )
        else:
            output, metadata = model(
                input_ids,
                attention_mask=attention_mask,
                force_iterations=5,  # Force 5 iterations
            )

            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                target_ids.view(-1),
                ignore_index=-100,
                reduction='sum'
            )

        # Count non-padding tokens
        valid_tokens = (target_ids != -100).sum().item()

        total_loss += loss.item()
        total_tokens += valid_tokens
        num_batches += 1

        # Track iteration statistics
        # FlashRecursiveTransformer returns 'iterations_per_position' and 'num_iterations'
        if 'iterations_per_position' in metadata:
            iters_per_pos = metadata['iterations_per_position']  # [batch, seq_len] float tensor
            # Average iterations per position (only count non-padding positions)
            mask = (target_ids != -100)
            if mask.any():
                avg_iters = (iters_per_pos * mask).sum().item() / mask.sum().item()
            else:
                avg_iters = iters_per_pos.mean().item()
            total_iterations += avg_iters

            # Count iteration distribution (round to int for bucketing)
            iters_int = iters_per_pos.long()
            for i in range(1, model.max_iterations + 1):
                count = ((iters_int == i) & mask).sum().item()
                iteration_counts[i] += count
        elif 'num_iterations' in metadata:
            # Fallback: use total iterations if per-position not available
            total_iterations += metadata['num_iterations']

        # Update progress bar
        running_ppl = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
        running_avg_iters = total_iterations / max(num_batches, 1)
        pbar.set_postfix({
            'ppl': f'{running_ppl:.2f}',
            'iters': f'{running_avg_iters:.2f}',
        })

    # Compute final metrics
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    avg_iterations = total_iterations / max(num_batches, 1)

    # Normalize iteration distribution
    total_positions = sum(iteration_counts.values())
    iteration_distribution = {
        k: v / max(total_positions, 1) * 100
        for k, v in sorted(iteration_counts.items())
    }

    metrics = {
        'loss': avg_loss,
        'perplexity': perplexity,
        'avg_iterations': avg_iterations,
        'total_tokens': total_tokens,
        'num_batches': num_batches,
        'iteration_distribution': iteration_distribution,
    }

    return metrics


def evaluate_per_iteration(
    model: FlashRecursiveTransformer,
    val_loader: DataLoader,
    device: str = 'cuda',
    use_amp: bool = True,
) -> Dict:
    """
    Evaluate model with each forced iteration count.

    Shows how performance improves with more iterations.
    """
    model.eval()

    results = {}

    for num_iters in range(1, model.max_iterations + 1):
        total_loss = 0.0
        total_tokens = 0

        print(f"\nEvaluating with {num_iters} iteration(s)...")

        for batch in tqdm(val_loader, desc=f"Iter {num_iters}", leave=False):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        output, _ = model(input_ids, force_iterations=num_iters)
                        loss = F.cross_entropy(
                            output.view(-1, output.size(-1)),
                            target_ids.view(-1),
                            ignore_index=-100,
                            reduction='sum'
                        )
                else:
                    output, _ = model(input_ids, force_iterations=num_iters)
                    loss = F.cross_entropy(
                        output.view(-1, output.size(-1)),
                        target_ids.view(-1),
                        ignore_index=-100,
                        reduction='sum'
                    )

            valid_tokens = (target_ids != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        results[num_iters] = {
            'loss': avg_loss,
            'perplexity': perplexity,
        }

        print(f"  Iterations: {num_iters} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

    return results


def print_results(metrics: Dict, per_iter_results: Optional[Dict] = None):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 40)
    print(f"{'Loss':<25} {metrics['loss']:>15.4f}")
    print(f"{'Perplexity':<25} {metrics['perplexity']:>15.2f}")
    print(f"{'Avg Iterations':<25} {metrics['avg_iterations']:>15.2f}")
    print(f"{'Total Tokens':<25} {metrics['total_tokens']:>15,}")
    print(f"{'Num Batches':<25} {metrics['num_batches']:>15,}")

    if metrics.get('iteration_distribution'):
        print("\n" + "-" * 40)
        print("Iteration Distribution:")
        for iters, pct in metrics['iteration_distribution'].items():
            bar = "█" * int(pct / 2)
            print(f"  {iters} iterations: {pct:>5.1f}% {bar}")

    if per_iter_results:
        print("\n" + "-" * 40)
        print("Performance by Forced Iteration Count:")
        print(f"  {'Iterations':<12} {'Loss':>10} {'Perplexity':>12}")
        print("  " + "-" * 34)
        for num_iters, result in per_iter_results.items():
            print(f"  {num_iters:<12} {result['loss']:>10.4f} {result['perplexity']:>12.2f}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Wikipedia model on validation dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_wikipedia.py                               # Auto-load best model
  python eval_wikipedia.py --checkpoint path/to/model.pt # Specific checkpoint
  python eval_wikipedia.py --val-samples 5000            # More validation samples
  python eval_wikipedia.py --dataset wikitext-103        # Use WikiText-103
  python eval_wikipedia.py --per-iteration               # Evaluate each iteration count
        """
    )

    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        default=None,
        help='Path to model checkpoint (auto-detects best model if not specified)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f'Directory to search for checkpoints (default: {DEFAULT_CHECKPOINT_DIR})'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['wikipedia-monthly', 'wikitext-103'],
        default='wikipedia-monthly',
        help='Validation dataset to use (default: wikipedia-monthly)'
    )
    parser.add_argument(
        '--val-samples',
        type=int,
        default=1000,
        help='Number of validation samples (default: 1000)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=16,
        help='Batch size for evaluation (default: 16)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: cuda if available)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable automatic mixed precision'
    )
    parser.add_argument(
        '--per-iteration',
        action='store_true',
        help='Also evaluate with each forced iteration count'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers (default: 4)'
    )

    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            print(f"Error: No checkpoint found in {args.checkpoint_dir}")
            print("Please train a model first using train_wikipedia.py or specify a checkpoint path.")
            return 1

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    print("=" * 60)
    print("Wikipedia Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Validation samples: {args.val_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"AMP: {not args.no_amp}")
    print("=" * 60)

    # Load model
    model, config = load_model(checkpoint_path, args.device)

    # Get tokenizer
    tokenizer = get_tokenizer("gpt2")

    # Create validation dataloader
    print(f"\nLoading {args.dataset} validation set...")

    if args.dataset == 'wikitext-103':
        val_loader = create_wikitext103_dataloader(
            tokenizer=tokenizer,
            max_seq_len=config.get('max_seq_len', 512),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:  # wikipedia-monthly
        val_dataset = WikipediaDatasetFinite(
            tokenizer=tokenizer,
            max_seq_len=config.get('max_seq_len', 512),
            max_samples=args.val_samples,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    print(f"Validation batches: {len(val_loader)}")

    # Main evaluation
    print("\n" + "-" * 60)
    print("Running evaluation (adaptive compute)...")
    print("-" * 60)

    metrics = evaluate(
        model=model,
        val_loader=val_loader,
        device=args.device,
        use_amp=not args.no_amp,
    )

    # Per-iteration evaluation
    per_iter_results = None
    if args.per_iteration:
        print("\n" + "-" * 60)
        print("Running per-iteration evaluation...")
        print("-" * 60)
        per_iter_results = evaluate_per_iteration(
            model=model,
            val_loader=val_loader,
            device=args.device,
            use_amp=not args.no_amp,
        )

    # Print results
    print_results(metrics, per_iter_results)

    return 0


if __name__ == '__main__':
    exit(main())
