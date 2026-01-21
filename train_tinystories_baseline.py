"""
Baseline training for TinyStories - NO recursive iterations.

This is the control experiment to verify that adaptive compute helps.
Uses the same architecture but:
- No cross-attention between iterations
- No done classifier
- Fixed number of layer passes (weight-tied)
- Standard LM loss only

Compare results with train_tinystories.py to isolate the effect of iterations.
"""

import os
import time
import argparse
from typing import Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from model import StandardWeightTiedTransformer
from dataset_tinystories import create_tinystories_dataloaders, get_tokenizer, TINYSTORIES_VOCAB_SIZE
from train_tinystories import (
    print_gpu_memory,
    find_max_batch_size,
    compute_batch_settings,
    _test_batch_size,
)


def compute_baseline_loss(
    output: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    """
    Standard language modeling loss - no done classifier, no iteration cost.
    """
    # Ignore index for padding
    ignore_idx = -100

    # Cross-entropy loss
    loss = F.cross_entropy(
        output.view(-1, output.size(-1)),
        target.view(-1),
        ignore_index=ignore_idx,
        reduction='mean'
    )

    # Compute metrics
    with torch.no_grad():
        perplexity = torch.exp(loss).item()

        # Token accuracy
        predictions = output.argmax(dim=-1)
        valid_mask = (target != ignore_idx)
        if valid_mask.any():
            accuracy = (predictions[valid_mask] == target[valid_mask]).float().mean().item()
        else:
            accuracy = 0.0

    metrics = {
        'task_loss': loss.item(),
        'perplexity': perplexity,
        'accuracy': accuracy,
    }

    return loss, metrics


class BaselineTrainer:
    """Training handler for baseline (non-recursive) model."""

    def __init__(
        self,
        model: StandardWeightTiedTransformer,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        accumulation_steps: int = 1
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.accumulation_steps = accumulation_steps

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Learning rate scheduler (cosine with warmup)
        total_steps = config.get('total_steps', 50000)
        warmup_steps = config.get('warmup_steps', 1000)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision
        self.use_amp = config.get('use_amp', True) and device == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Tracking
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)
        self.global_step = 0

    def train_step_accumulated(self, train_iter) -> Tuple[Dict, any]:
        """Training step with gradient accumulation."""
        self.model.train()
        self.optimizer.zero_grad()

        accumulated_metrics = defaultdict(float)
        accumulated_loss = 0.0

        for accum_idx in range(self.accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            if self.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    output, metadata = self.model(input_ids)
                    loss, metrics = compute_baseline_loss(output, target_ids)
                    scaled_loss = loss / self.accumulation_steps

                self.scaler.scale(scaled_loss).backward()
            else:
                output, metadata = self.model(input_ids)
                loss, metrics = compute_baseline_loss(output, target_ids)
                scaled_loss = loss / self.accumulation_steps
                scaled_loss.backward()

            accumulated_loss += loss.item()
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    accumulated_metrics[k] += v

        # Optimizer step
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1

        final_metrics = {k: v / self.accumulation_steps for k, v in accumulated_metrics.items()}
        final_metrics['loss'] = accumulated_loss / self.accumulation_steps
        final_metrics['lr'] = self.scheduler.get_last_lr()[0]

        return final_metrics, train_iter

    @torch.no_grad()
    def validate(self) -> Dict:
        """Run validation."""
        self.model.eval()

        total_loss = 0.0
        total_metrics = defaultdict(float)
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            output, metadata = self.model(input_ids)
            loss, metrics = compute_baseline_loss(output, target_ids)

            total_loss += loss.item()
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    total_metrics[k] += v
            num_batches += 1

        total_loss /= num_batches
        for k in total_metrics:
            total_metrics[k] /= num_batches

        return {'loss': total_loss, **total_metrics}

    def train(
        self,
        total_steps: int,
        save_dir: str = 'checkpoints_tinystories_baseline',
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000
    ):
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)

        actual_batch_size = self.config.get('actual_batch_size', self.config.get('batch_size', 64))
        effective_batch_size = actual_batch_size * self.accumulation_steps

        print(f"\nStarting BASELINE training for {total_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {actual_batch_size}")
        print(f"Accumulation steps: {self.accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Layer repeats: {self.model.n_repeats} (fixed, no adaptive compute)")
        print_gpu_memory()
        print("-" * 60)

        train_iter = iter(self.train_loader)
        running_metrics = defaultdict(float)
        running_count = 0
        start_time = time.time()

        while self.global_step < total_steps:
            metrics, train_iter = self.train_step_accumulated(train_iter)

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running_metrics[k] += v
            running_count += 1

            if self.global_step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = log_interval / elapsed if elapsed > 0 else 0
                samples_per_sec = steps_per_sec * effective_batch_size

                avg_metrics = {k: v / running_count for k, v in running_metrics.items()}

                print(f"Step {self.global_step}/{total_steps} | "
                      f"Loss: {avg_metrics['loss']:.4f} | "
                      f"PPL: {avg_metrics['perplexity']:.2f} | "
                      f"Acc: {avg_metrics['accuracy']:.3f} | "
                      f"LR: {avg_metrics['lr']:.2e} | "
                      f"{steps_per_sec:.1f} steps/s ({samples_per_sec:.0f} samples/s)")

                for k, v in avg_metrics.items():
                    self.history[f'train_{k}'].append(v)
                self.history['step'].append(self.global_step)

                running_metrics = defaultdict(float)
                running_count = 0
                start_time = time.time()

            if self.global_step % eval_interval == 0:
                val_metrics = self.validate()
                print(f"  [Val] Loss: {val_metrics['loss']:.4f} | "
                      f"PPL: {val_metrics['perplexity']:.2f} | "
                      f"Acc: {val_metrics['accuracy']:.3f}")

                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        self.history[f'val_{k}'].append(v)

                print_gpu_memory("  ")

                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'))
                    print(f"  New best model saved!")

                print()

            if self.global_step % save_interval == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_{self.global_step}.pt'))

        self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'))

        print("-" * 60)
        print("BASELINE Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation PPL: {torch.exp(torch.tensor(self.best_val_loss)).item():.2f}")
        print_gpu_memory("Final ")

        return dict(self.history)

    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history),
            'global_step': self.global_step,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = defaultdict(list, checkpoint.get('history', {}))
        self.global_step = checkpoint.get('global_step', 0)
        return self.global_step


def _test_batch_size_baseline(
    model: StandardWeightTiedTransformer,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    device: str,
    gpu_mem_gb: float,
    use_amp: bool = True
) -> Tuple[bool, float]:
    """Test if a batch size fits for baseline model."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        if use_amp:
            with autocast('cuda', dtype=torch.bfloat16):
                output, metadata = model(dummy_input)
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)),
                    dummy_target.view(-1),
                    reduction='mean'
                )
        else:
            output, metadata = model(dummy_input)
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                dummy_target.view(-1),
                reduction='mean'
            )

        loss.backward()
        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        model.zero_grad(set_to_none=True)
        del dummy_input, dummy_target, output, metadata, loss
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if peak_memory > gpu_mem_gb * 0.95:
            return False, peak_memory

        return True, peak_memory

    except RuntimeError as e:
        error_str = str(e).lower()
        if "out of memory" in error_str or "cuda" in error_str or "memory" in error_str:
            try:
                model.zero_grad(set_to_none=True)
            except:
                pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return False, 0.0
        else:
            raise e


def find_max_batch_size_baseline(
    model: StandardWeightTiedTransformer,
    vocab_size: int,
    seq_len: int,
    device: str,
    start: int = 8,
    max_batch: int = 512,
    use_amp: bool = True
) -> int:
    """Binary search for max batch size (baseline model version)."""
    if device != 'cuda':
        return start

    model.train()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Finding optimal batch size (GPU: {gpu_mem_gb:.1f}GB)...")

    low = start
    high = start

    print("  Phase 1: Finding upper bound...")
    while high <= max_batch:
        success, peak = _test_batch_size_baseline(model, vocab_size, seq_len, high, device, gpu_mem_gb, use_amp)
        if success:
            print(f"    batch_size={high}... OK (peak: {peak:.1f}GB)")
            low = high
            high *= 2
        else:
            if peak > 0:
                print(f"    batch_size={high}... OOM (peak: {peak:.1f}GB)")
            else:
                print(f"    batch_size={high}... OOM")
            break

    if high > max_batch:
        high = max_batch
        success, peak = _test_batch_size_baseline(model, vocab_size, seq_len, high, device, gpu_mem_gb, use_amp)
        if success:
            print(f"    batch_size={high}... OK (peak: {peak:.1f}GB)")
            print(f"  Max batch size: {high} (capped)")
            return high

    if high > low:
        print(f"  Phase 2: Binary search between {low} and {high}...")
        while high - low > max(1, low // 8):
            mid = (low + high) // 2
            if mid == low:
                break
            success, peak = _test_batch_size_baseline(model, vocab_size, seq_len, mid, device, gpu_mem_gb, use_amp)
            if success:
                print(f"    batch_size={mid}... OK (peak: {peak:.1f}GB)")
                low = mid
            else:
                if peak > 0:
                    print(f"    batch_size={mid}... OOM (peak: {peak:.1f}GB)")
                else:
                    print(f"    batch_size={mid}... OOM")
                high = mid

    if low == 0:
        return start

    print(f"  Max batch size: {low}")
    return low


def train_baseline(config: Dict = None, resume_from: str = None):
    """Main training function for baseline model."""
    if config is None:
        config = {}

    print("=" * 60)
    print("BASELINE MODEL (No Adaptive Compute)")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
        print_gpu_memory()
    else:
        print("WARNING: CUDA not available!")
        device = 'cpu'

    tokenizer = get_tokenizer(config.get('tokenizer_name', 'EleutherAI/gpt-neo-125M'))

    # Create baseline model - single pass through layers (no repeats)
    n_repeats = config.get('n_repeats', 1)

    model = StandardWeightTiedTransformer(
        vocab_size=config.get('vocab_size', TINYSTORIES_VOCAB_SIZE),
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        n_repeats=n_repeats,  # Same as max_iterations for fair comparison
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 256)
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Layer repeats: {n_repeats} (fixed)")

    model = model.to(device)

    # Find optimal batch size
    if config.get('auto_batch_size', True) and device == 'cuda':
        max_batch = find_max_batch_size_baseline(
            model=model,
            vocab_size=config.get('vocab_size', TINYSTORIES_VOCAB_SIZE),
            seq_len=config.get('max_seq_len', 256),
            device=device,
            use_amp=config.get('use_amp', True)
        )
        actual_batch_size = max(1, int(max_batch * config.get('batch_size_safety_margin', 0.9)))
        print(f"Using batch_size={actual_batch_size} ({config.get('batch_size_safety_margin', 0.9)*100:.0f}% safety margin)")
    else:
        actual_batch_size = config.get('batch_size', 64)

    target_effective = config.get('target_effective_batch_size', 128)
    accumulation_steps = max(1, target_effective // actual_batch_size)
    effective_batch_size = actual_batch_size * accumulation_steps

    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    config['actual_batch_size'] = actual_batch_size
    config['accumulation_steps'] = accumulation_steps

    train_loader, val_loader, tokenizer = create_tinystories_dataloaders(
        tokenizer_name=config.get('tokenizer_name', 'EleutherAI/gpt-neo-125M'),
        max_seq_len=config.get('max_seq_len', 256),
        batch_size=actual_batch_size,
        num_workers=0,
        val_samples=config.get('val_samples', 1000)
    )

    trainer = BaselineTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        accumulation_steps=accumulation_steps
    )

    if resume_from is not None:
        print(f"Resuming from {resume_from}...")
        trainer.load_checkpoint(resume_from)
        print(f"Resumed at step {trainer.global_step}")

    history = trainer.train(
        total_steps=config.get('total_steps', 50000),
        save_dir=config.get('save_dir', 'checkpoints_tinystories_baseline'),
        log_interval=config.get('log_interval', 100),
        eval_interval=config.get('eval_interval', 1000),
        save_interval=config.get('save_interval', 5000)
    )

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BASELINE model on TinyStories (no adaptive compute)')
    parser.add_argument('--steps', type=int, default=50000, help='Total training steps')
    parser.add_argument('--batch-size', type=int, default=64, help='Manual batch size')
    parser.add_argument('--target-batch-size', type=int, default=128, help='Target effective batch size')
    parser.add_argument('--no-auto-batch', action='store_true', help='Disable auto batch size')
    parser.add_argument('--d-model', type=int, default=864, help='Model dimension (864 gives ~80M params)')
    parser.add_argument('--n-layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--n-repeats', type=int, default=1, help='Number of layer repeats (1 = single pass)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default='checkpoints_tinystories_baseline', help='Save directory')

    args = parser.parse_args()

    # Compute n_heads ensuring d_model is divisible
    n_heads = 8 if args.d_model % 8 == 0 else (6 if args.d_model % 6 == 0 else 4)

    config = {
        'vocab_size': TINYSTORIES_VOCAB_SIZE,  # 10K (like TinyStories paper)
        'd_model': args.d_model,
        'n_heads': n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_model * 4,
        'n_repeats': args.n_repeats,  # 1 = single pass (no repeats)
        'dropout': 0.1,
        'max_seq_len': 256,

        'batch_size': args.batch_size,
        'target_effective_batch_size': args.target_batch_size,
        'auto_batch_size': not args.no_auto_batch,
        'batch_size_safety_margin': 0.9,

        'total_steps': args.steps,
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'use_amp': True,

        'save_dir': args.save_dir,
        'log_interval': 100,
        'eval_interval': 1000,
        'save_interval': 5000,
        'val_samples': 1000,
    }

    train_baseline(config, resume_from=args.resume)
