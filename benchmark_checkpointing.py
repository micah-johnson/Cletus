"""
Benchmark comparing training performance with/without gradient checkpointing.

Measures:
- Peak GPU memory usage
- Training throughput (samples/sec, tokens/sec)
- Forward/backward pass times
- Loss values (to verify correctness)
"""

import os
import time
import argparse
from typing import Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler

from model import RecursiveTransformer, RecursiveTransformerLayer
from dataset_tinystories import get_tokenizer, TINYSTORIES_VOCAB_SIZE


class CheckpointedRecursiveTransformerLayer(RecursiveTransformerLayer):
    """
    Wrapper that applies gradient checkpointing to the layer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_checkpointing = False

    def forward(self, x, previous_states=None, attn_mask=None, return_attention=False):
        if self.use_checkpointing and self.training:
            # Checkpoint the forward pass
            return checkpoint(
                super().forward,
                x, previous_states, attn_mask, return_attention,
                use_reentrant=False
            )
        return super().forward(x, previous_states, attn_mask, return_attention)


def create_model(d_model: int, n_layers: int, max_iters: int, use_checkpointing: bool) -> RecursiveTransformer:
    """Create model, optionally with gradient checkpointing."""
    n_heads = 8 if d_model % 8 == 0 else 4

    model = RecursiveTransformer(
        vocab_size=TINYSTORIES_VOCAB_SIZE,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
        max_iterations=max_iters,
        dropout=0.1,
        max_seq_len=256
    )

    if use_checkpointing:
        # Replace layers with checkpointed versions
        new_layers = nn.ModuleList()
        for layer in model.layers:
            ckpt_layer = CheckpointedRecursiveTransformerLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_model * 4,
                dropout=0.1,
                max_seq_len=256
            )
            # Copy weights
            ckpt_layer.load_state_dict(layer.state_dict())
            ckpt_layer.use_checkpointing = True
            new_layers.append(ckpt_layer)
        model.layers = new_layers

    return model


def benchmark_single_config(
    model: RecursiveTransformer,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    use_amp: bool,
    device: str,
    warmup_steps: int = 3
) -> Dict:
    """
    Run benchmark for a single configuration.

    Returns dict with timing and memory metrics.
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler('cuda') if use_amp and device == 'cuda' else None

    # Create dummy data
    input_ids = torch.randint(0, TINYSTORIES_VOCAB_SIZE, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, TINYSTORIES_VOCAB_SIZE, (batch_size, seq_len), device=device)

    # Clear memory stats
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    forward_times = []
    backward_times = []
    total_times = []
    losses = []

    for step in range(num_steps + warmup_steps):
        optimizer.zero_grad()

        # Forward
        if device == 'cuda':
            torch.cuda.synchronize()
        fwd_start = time.perf_counter()

        if use_amp and device == 'cuda':
            with autocast('cuda', dtype=torch.bfloat16):
                output, metadata = model(input_ids, force_iterations=model.max_iterations)
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)),
                    target_ids.view(-1),
                    reduction='mean'
                )
        else:
            output, metadata = model(input_ids, force_iterations=model.max_iterations)
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                target_ids.view(-1),
                reduction='mean'
            )

        if device == 'cuda':
            torch.cuda.synchronize()
        fwd_end = time.perf_counter()

        # Backward
        bwd_start = time.perf_counter()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if device == 'cuda':
            torch.cuda.synchronize()
        bwd_end = time.perf_counter()

        # Record (skip warmup)
        if step >= warmup_steps:
            forward_times.append(fwd_end - fwd_start)
            backward_times.append(bwd_end - bwd_start)
            total_times.append(bwd_end - fwd_start)
            losses.append(loss.item())

    # Get memory stats
    if device == 'cuda':
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        reserved_memory_gb = torch.cuda.max_memory_reserved() / 1024**3
    else:
        peak_memory_gb = 0
        reserved_memory_gb = 0

    # Compute stats
    avg_forward = sum(forward_times) / len(forward_times)
    avg_backward = sum(backward_times) / len(backward_times)
    avg_total = sum(total_times) / len(total_times)

    samples_per_sec = batch_size / avg_total
    tokens_per_sec = batch_size * seq_len / avg_total

    return {
        'avg_forward_ms': avg_forward * 1000,
        'avg_backward_ms': avg_backward * 1000,
        'avg_total_ms': avg_total * 1000,
        'samples_per_sec': samples_per_sec,
        'tokens_per_sec': tokens_per_sec,
        'peak_memory_gb': peak_memory_gb,
        'reserved_memory_gb': reserved_memory_gb,
        'avg_loss': sum(losses) / len(losses),
        'final_loss': losses[-1]
    }


def find_max_batch_size(
    model: RecursiveTransformer,
    seq_len: int,
    device: str,
    use_amp: bool,
    start: int = 4,
    max_batch: int = 256
) -> int:
    """Binary search for max batch size that fits in memory."""
    if device != 'cuda':
        return start

    model = model.to(device)
    model.train()

    low, high = start, start

    # Find upper bound
    while high <= max_batch:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            input_ids = torch.randint(0, TINYSTORIES_VOCAB_SIZE, (high, seq_len), device=device)
            target_ids = torch.randint(0, TINYSTORIES_VOCAB_SIZE, (high, seq_len), device=device)

            if use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    output, _ = model(input_ids, force_iterations=model.max_iterations)
                    loss = F.cross_entropy(output.view(-1, output.size(-1)), target_ids.view(-1))
            else:
                output, _ = model(input_ids, force_iterations=model.max_iterations)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), target_ids.view(-1))

            loss.backward()
            torch.cuda.synchronize()

            del input_ids, target_ids, output, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            low = high
            high *= 2

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                break
            raise

    # Binary search
    high = min(high, max_batch)
    while high - low > 1:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()

            input_ids = torch.randint(0, TINYSTORIES_VOCAB_SIZE, (mid, seq_len), device=device)
            target_ids = torch.randint(0, TINYSTORIES_VOCAB_SIZE, (mid, seq_len), device=device)

            if use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    output, _ = model(input_ids, force_iterations=model.max_iterations)
                    loss = F.cross_entropy(output.view(-1, output.size(-1)), target_ids.view(-1))
            else:
                output, _ = model(input_ids, force_iterations=model.max_iterations)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), target_ids.view(-1))

            loss.backward()
            torch.cuda.synchronize()

            del input_ids, target_ids, output, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

            low = mid
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                high = mid
            else:
                raise

    return low


def run_benchmark(args):
    """Run full benchmark comparing checkpointing vs no checkpointing."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("GRADIENT CHECKPOINTING BENCHMARK")
    print("=" * 70)
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Model: d_model={args.d_model}, n_layers={args.n_layers}, max_iters={args.max_iters}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Mixed precision (AMP): {args.use_amp}")
    print(f"Benchmark steps: {args.num_steps}")
    print("=" * 70)

    results = {}

    for use_ckpt in [False, True]:
        config_name = "WITH checkpointing" if use_ckpt else "WITHOUT checkpointing"
        print(f"\n--- {config_name} ---")

        # Create fresh model
        model = create_model(args.d_model, args.n_layers, args.max_iters, use_ckpt)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")

        # Find max batch size
        print("Finding max batch size...")
        if args.batch_size > 0:
            batch_size = args.batch_size
            print(f"Using fixed batch size: {batch_size}")
        else:
            batch_size = find_max_batch_size(
                model, args.seq_len, device, args.use_amp,
                start=4, max_batch=256
            )
            print(f"Max batch size: {batch_size}")

        # Run benchmark
        print(f"Running benchmark with batch_size={batch_size}...")
        torch.cuda.empty_cache() if device == 'cuda' else None

        metrics = benchmark_single_config(
            model=model,
            batch_size=batch_size,
            seq_len=args.seq_len,
            num_steps=args.num_steps,
            use_amp=args.use_amp,
            device=device,
            warmup_steps=3
        )

        metrics['batch_size'] = batch_size
        results[use_ckpt] = metrics

        print(f"  Peak Memory: {metrics['peak_memory_gb']:.2f} GB")
        print(f"  Forward:  {metrics['avg_forward_ms']:.1f} ms")
        print(f"  Backward: {metrics['avg_backward_ms']:.1f} ms")
        print(f"  Total:    {metrics['avg_total_ms']:.1f} ms")
        print(f"  Throughput: {metrics['samples_per_sec']:.1f} samples/sec, {metrics['tokens_per_sec']:.0f} tokens/sec")
        print(f"  Avg Loss: {metrics['avg_loss']:.4f}")

        # Cleanup
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    no_ckpt = results[False]
    with_ckpt = results[True]

    print(f"\n{'Metric':<25} {'No Ckpt':>15} {'With Ckpt':>15} {'Change':>15}")
    print("-" * 70)

    def fmt_change(no_val, with_val, lower_is_better=True):
        diff = with_val - no_val
        pct = (diff / no_val) * 100 if no_val != 0 else 0
        sign = "+" if diff > 0 else ""
        color = "better" if (diff < 0 and lower_is_better) or (diff > 0 and not lower_is_better) else "worse"
        return f"{sign}{pct:.1f}% ({color})"

    print(f"{'Batch Size':<25} {no_ckpt['batch_size']:>15} {with_ckpt['batch_size']:>15} {fmt_change(no_ckpt['batch_size'], with_ckpt['batch_size'], lower_is_better=False)}")
    print(f"{'Peak Memory (GB)':<25} {no_ckpt['peak_memory_gb']:>15.2f} {with_ckpt['peak_memory_gb']:>15.2f} {fmt_change(no_ckpt['peak_memory_gb'], with_ckpt['peak_memory_gb'], lower_is_better=True)}")
    print(f"{'Forward (ms)':<25} {no_ckpt['avg_forward_ms']:>15.1f} {with_ckpt['avg_forward_ms']:>15.1f} {fmt_change(no_ckpt['avg_forward_ms'], with_ckpt['avg_forward_ms'], lower_is_better=True)}")
    print(f"{'Backward (ms)':<25} {no_ckpt['avg_backward_ms']:>15.1f} {with_ckpt['avg_backward_ms']:>15.1f} {fmt_change(no_ckpt['avg_backward_ms'], with_ckpt['avg_backward_ms'], lower_is_better=True)}")
    print(f"{'Total (ms)':<25} {no_ckpt['avg_total_ms']:>15.1f} {with_ckpt['avg_total_ms']:>15.1f} {fmt_change(no_ckpt['avg_total_ms'], with_ckpt['avg_total_ms'], lower_is_better=True)}")
    print(f"{'Samples/sec':<25} {no_ckpt['samples_per_sec']:>15.1f} {with_ckpt['samples_per_sec']:>15.1f} {fmt_change(no_ckpt['samples_per_sec'], with_ckpt['samples_per_sec'], lower_is_better=False)}")
    print(f"{'Tokens/sec':<25} {no_ckpt['tokens_per_sec']:>15.0f} {with_ckpt['tokens_per_sec']:>15.0f} {fmt_change(no_ckpt['tokens_per_sec'], with_ckpt['tokens_per_sec'], lower_is_better=False)}")

    # Memory savings at same batch size
    if no_ckpt['batch_size'] == with_ckpt['batch_size']:
        mem_savings = (1 - with_ckpt['peak_memory_gb'] / no_ckpt['peak_memory_gb']) * 100
        print(f"\nMemory savings at same batch size: {mem_savings:.1f}%")

    # Effective throughput comparison (considering larger batch sizes possible with checkpointing)
    if with_ckpt['batch_size'] > no_ckpt['batch_size']:
        print(f"\nCheckpointing allows {with_ckpt['batch_size'] / no_ckpt['batch_size']:.1f}x larger batch size!")
        print(f"Effective throughput improvement: {with_ckpt['tokens_per_sec'] / no_ckpt['tokens_per_sec']:.2f}x")

    print("\n" + "=" * 70)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark gradient checkpointing')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--max-iters', type=int, default=6, help='Max recursive iterations')
    parser.add_argument('--seq-len', type=int, default=256, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=0, help='Fixed batch size (0=auto)')
    parser.add_argument('--num-steps', type=int, default=20, help='Number of benchmark steps')
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')

    args = parser.parse_args()
    args.use_amp = not args.no_amp

    run_benchmark(args)
