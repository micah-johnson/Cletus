"""
Training loop for 125M Recursive Transformer on Wikipedia.

Key features:
- 125M parameter FlashRecursiveTransformer (d_model=704, n_layers=9)
- Max 5 iterations with adaptive compute
- Wikipedia dataset (omarkamali/wikipedia-monthly)
- GPT-2 tokenizer with full 50K vocab
- Automatic batch size detection
- Gradient accumulation for target effective batch size
"""

import os
import time
import random
import argparse
from typing import Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from model import FlashRecursiveTransformer
from dataset_wikipedia import create_wikipedia_dataloaders, get_tokenizer, GPT2_VOCAB_SIZE
from train_tinystories import (
    print_gpu_memory,
    compute_lm_loss,
    enable_gradient_checkpointing,
)


# =============================================================================
# 125M Model Config
# =============================================================================

MODEL_125M_CONFIG = {
    'vocab_size': GPT2_VOCAB_SIZE,  # 50257 (full GPT-2 vocab)
    'd_model': 704,
    'n_heads': 8,
    'n_layers': 9,
    'd_ff': 2816,  # 704 * 4
    'max_iterations': 5,
    'dropout': 0.1,
    'max_seq_len': 512,
}


# =============================================================================
# Batch Size Detection (adapted for Wikipedia)
# =============================================================================

def _test_batch_size(
    model: nn.Module,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    device: str,
    gpu_mem_gb: float,
    use_amp: bool = True
) -> Tuple[bool, float]:
    """Test if a batch size fits in memory."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        if use_amp:
            with autocast('cuda', dtype=torch.bfloat16):
                output, metadata = model(dummy_input, force_iterations=model.max_iterations)
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)),
                    dummy_target.view(-1),
                    reduction='mean'
                )
        else:
            output, metadata = model(dummy_input, force_iterations=model.max_iterations)
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


def find_max_batch_size(
    model: nn.Module,
    vocab_size: int,
    seq_len: int,
    device: str,
    start: int = 4,
    max_batch: int = 256,
    use_amp: bool = True
) -> int:
    """Binary search for max batch size."""
    if device != 'cuda':
        return start

    model.train()
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Finding optimal batch size (GPU: {gpu_mem_gb:.1f}GB)...")

    low = start
    high = start

    print("  Phase 1: Finding upper bound...")
    while high <= max_batch:
        success, peak = _test_batch_size(model, vocab_size, seq_len, high, device, gpu_mem_gb, use_amp)
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
        success, peak = _test_batch_size(model, vocab_size, seq_len, high, device, gpu_mem_gb, use_amp)
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
            success, peak = _test_batch_size(model, vocab_size, seq_len, mid, device, gpu_mem_gb, use_amp)
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


# =============================================================================
# Trainer Class
# =============================================================================

class WikipediaTrainer:
    """Training handler for Wikipedia language modeling."""

    def __init__(
        self,
        model: FlashRecursiveTransformer,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        accumulation_steps: int = 1,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.accumulation_steps = accumulation_steps

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.1),
            betas=(0.9, 0.95),
        )

        # Learning rate scheduler (cosine with warmup)
        total_steps = config.get('total_steps', 100000)
        warmup_steps = config.get('warmup_steps', 2000)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Mixed precision
        self.use_amp = config.get('use_amp', True) and device == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Loss weights
        self.iteration_cost = config.get('iteration_cost', 0.01)
        self.done_supervision_weight = config.get('done_supervision_weight', 0.5)

        # Tracking
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)
        self.global_step = 0

    def train_step_accumulated(self, train_iter, random_max_iters: bool = True) -> Tuple[Dict, any]:
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

            # Randomize iterations for adaptive compute learning
            if random_max_iters:
                batch_max_iters = random.randint(1, self.model.max_iterations)
            else:
                batch_max_iters = self.model.max_iterations

            if self.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    output, metadata = self.model(input_ids, force_iterations=batch_max_iters)
                    loss, metrics = compute_lm_loss(
                        output, target_ids, metadata,
                        iteration_cost=self.iteration_cost,
                        done_supervision_weight=self.done_supervision_weight
                    )
                    scaled_loss = loss / self.accumulation_steps

                self.scaler.scale(scaled_loss).backward()
            else:
                output, metadata = self.model(input_ids, force_iterations=batch_max_iters)
                loss, metrics = compute_lm_loss(
                    output, target_ids, metadata,
                    iteration_cost=self.iteration_cost,
                    done_supervision_weight=self.done_supervision_weight
                )
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

            output, metadata = self.model(
                input_ids,
                force_iterations=self.model.max_iterations,
                threshold=0.7
            )
            loss, metrics = compute_lm_loss(
                output, target_ids, metadata,
                iteration_cost=self.iteration_cost,
                done_supervision_weight=self.done_supervision_weight
            )

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
        save_dir: str = 'checkpoints_wikipedia_125m',
        log_interval: int = 100,
        eval_interval: int = 2000,
        save_interval: int = 10000
    ):
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)

        actual_batch_size = self.config.get('actual_batch_size', self.config.get('batch_size', 32))
        effective_batch_size = actual_batch_size * self.accumulation_steps

        print(f"\nStarting Wikipedia 125M training for {total_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Model: FlashRecursiveTransformer (125M params)")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {actual_batch_size}")
        print(f"Accumulation steps: {self.accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Max iterations: {self.model.max_iterations}")
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
                      f"Avg Iters: {avg_metrics.get('avg_iterations', 0):.2f} | "
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
                      f"Avg Iters: {val_metrics.get('avg_iterations', 0):.2f}")

                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        self.history[f'val_{k}'].append(v)

                # Print gate values
                if hasattr(self.model, 'get_gate_values'):
                    gates = self.model.get_gate_values()
                    print(f"  Cross-attn gates: {[f'{g:.3f}' for g in gates]}")
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
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
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


# =============================================================================
# Main Training Function
# =============================================================================

def train_wikipedia_125m(config: Dict = None, resume_from: str = None):
    """Main training function for 125M Wikipedia model."""
    if config is None:
        config = {}

    print("=" * 60)
    print("125M FlashRecursiveTransformer on Wikipedia")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
        print_gpu_memory()
    else:
        print("WARNING: CUDA not available!")
        device = 'cpu'

    # Merge with default 125M config
    full_config = {**MODEL_125M_CONFIG, **config}

    # Create model
    model = FlashRecursiveTransformer(
        vocab_size=full_config['vocab_size'],
        d_model=full_config['d_model'],
        n_heads=full_config['n_heads'],
        n_layers=full_config['n_layers'],
        d_ff=full_config['d_ff'],
        max_iterations=full_config['max_iterations'],
        dropout=full_config['dropout'],
        max_seq_len=full_config['max_seq_len']
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device)

    # Enable gradient checkpointing if requested
    if full_config.get('gradient_checkpointing', False):
        enable_gradient_checkpointing(model)

    # Find optimal batch size
    if full_config.get('auto_batch_size', True) and device == 'cuda':
        max_batch = find_max_batch_size(
            model=model,
            vocab_size=full_config['vocab_size'],
            seq_len=full_config['max_seq_len'],
            device=device,
            start=4,
            max_batch=256,
            use_amp=full_config.get('use_amp', True)
        )
        actual_batch_size = max(1, int(max_batch * full_config.get('batch_size_safety_margin', 0.8)))
        print(f"Using batch_size={actual_batch_size} (80% safety margin)")
    else:
        actual_batch_size = full_config.get('batch_size', 32)

    target_effective = full_config.get('target_effective_batch_size', 256)
    accumulation_steps = max(1, target_effective // actual_batch_size)
    effective_batch_size = actual_batch_size * accumulation_steps

    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    full_config['actual_batch_size'] = actual_batch_size
    full_config['accumulation_steps'] = accumulation_steps

    # Create dataloaders
    train_loader, val_loader, tokenizer = create_wikipedia_dataloaders(
        tokenizer_name="gpt2",
        max_seq_len=full_config['max_seq_len'],
        batch_size=actual_batch_size,
        num_workers=0,
        val_samples=full_config.get('val_samples', 1000),
        dataset_name=full_config.get('dataset_name', "omarkamali/wikipedia-monthly"),
        dataset_config=full_config.get('dataset_config', "20251201.en"),
    )

    # Create trainer
    trainer = WikipediaTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=full_config,
        device=device,
        accumulation_steps=accumulation_steps,
    )

    if resume_from is not None:
        print(f"Resuming from {resume_from}...")
        trainer.load_checkpoint(resume_from)
        print(f"Resumed at step {trainer.global_step}")

    history = trainer.train(
        total_steps=full_config.get('total_steps', 100000),
        save_dir=full_config.get('save_dir', 'checkpoints_wikipedia_125m'),
        log_interval=full_config.get('log_interval', 100),
        eval_interval=full_config.get('eval_interval', 2000),
        save_interval=full_config.get('save_interval', 10000)
    )

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 125M Recursive Transformer on Wikipedia')
    parser.add_argument('--steps', type=int, default=100000, help='Total training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Manual batch size')
    parser.add_argument('--target-batch-size', type=int, default=256, help='Target effective batch size')
    parser.add_argument('--no-auto-batch', action='store_true', help='Disable auto batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max-iters', type=int, default=5, help='Max recursive iterations')
    parser.add_argument('--seq-len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default='checkpoints_wikipedia_125m', help='Save directory')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Enable gradient checkpointing')

    args = parser.parse_args()

    config = {
        # Training params
        'total_steps': args.steps,
        'learning_rate': args.lr,
        'weight_decay': 0.1,
        'warmup_steps': 2000,
        'use_amp': True,

        # Batch size
        'batch_size': args.batch_size,
        'target_effective_batch_size': args.target_batch_size,
        'auto_batch_size': not args.no_auto_batch,
        'batch_size_safety_margin': 0.8,

        # Model overrides
        'max_iterations': args.max_iters,
        'max_seq_len': args.seq_len,
        'gradient_checkpointing': args.gradient_checkpointing,

        # Loss weights
        'iteration_cost': 0.01,
        'done_supervision_weight': 0.5,

        # Logging
        'save_dir': args.save_dir,
        'log_interval': 100,
        'eval_interval': 2000,
        'save_interval': 10000,
        'val_samples': 1000,
    }

    train_wikipedia_125m(config, resume_from=args.resume)
