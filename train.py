"""
Training loop for 125M Recursive Transformer on OpenWebText.

Key features:
- 125M parameter FlashRecursiveTransformer
- Exactly 1 epoch through OpenWebText (~8M documents, ~4B tokens)
- FineWeb validation set
- Automatic batch size detection
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
from dataset_openwebtext import (
    create_openwebtext_dataloaders,
    get_tokenizer,
    GPT2_VOCAB_SIZE,
    estimate_openwebtext_tokens,
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
# Utilities
# =============================================================================

def print_gpu_memory(prefix: str = ""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"{prefix}GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")


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
                output, metadata = model(dummy_input, max_iters=model.max_iterations)
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)),
                    dummy_target.view(-1),
                    reduction='mean'
                )
        else:
            output, metadata = model(dummy_input, max_iters=model.max_iterations)
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
# Loss Function
# =============================================================================

def compute_lm_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    metadata: Dict,
    iteration_cost: float = 0.01,
    done_supervision_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict]:
    """
    Language modeling loss with done classifier supervision.

    Done classifier learns when each position's prediction is correct.
    """
    device = output.device
    batch_size, seq_len = output.size(0), output.size(1)
    all_outputs = metadata['all_outputs']
    done_logits = metadata['done_logits']
    done_probs = metadata['done_probs']
    num_iters = len(all_outputs)

    # Ignore index for padding
    ignore_idx = -100

    # 1. Task loss on final output
    task_loss = F.cross_entropy(
        output.view(-1, output.size(-1)),
        target.view(-1),
        ignore_index=ignore_idx,
        reduction='mean'
    )

    # 2. Per-position done classifier supervision
    per_position_correct = []
    per_iter_accuracies = []

    for iter_idx, iter_output in enumerate(all_outputs):
        predictions = iter_output.argmax(dim=-1)
        correct = (predictions == target).float()

        valid_mask = (target != ignore_idx)
        if valid_mask.any():
            acc = correct[valid_mask].mean().item()
        else:
            acc = 0.0

        per_position_correct.append(correct)
        per_iter_accuracies.append(acc)

    # Stack: [batch, num_iters, seq_len]
    done_targets = torch.stack(per_position_correct, dim=1)

    # Cumulative max: once correct, stays "done"
    done_targets_cummax, _ = torch.cummax(done_targets, dim=1)

    # Valid positions mask
    valid_mask_single = (target != ignore_idx)
    valid_mask = valid_mask_single.unsqueeze(1).expand(-1, num_iters, -1)

    # BCE loss
    done_supervision_loss = F.binary_cross_entropy_with_logits(
        done_logits,
        done_targets_cummax,
        reduction='none'
    )
    done_supervision_loss = (done_supervision_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

    # 3. Iteration cost
    continuation_probs = 1 - done_probs
    expected_iters_per_pos = continuation_probs.sum(dim=1)

    if valid_mask_single.any():
        iter_loss = iteration_cost * expected_iters_per_pos[valid_mask_single].mean()
    else:
        iter_loss = torch.tensor(0.0, device=device)

    total_loss = task_loss + done_supervision_weight * done_supervision_loss + iter_loss

    with torch.no_grad():
        perplexity = torch.exp(task_loss).item()

        if 'iterations_per_position' in metadata:
            iters_per_pos = metadata['iterations_per_position']
            if valid_mask_single.any():
                avg_iters = iters_per_pos[valid_mask_single].mean().item()
            else:
                avg_iters = num_iters
        else:
            avg_iters = num_iters

    metrics = {
        'task_loss': task_loss.item(),
        'done_loss': done_supervision_loss.item(),
        'iter_loss': iter_loss.item() if isinstance(iter_loss, torch.Tensor) else iter_loss,
        'perplexity': perplexity,
        'avg_iterations': avg_iters,
        'max_iterations_run': num_iters,
        'mean_done_prob': done_probs[:, -1, :].mean().item(),
        'per_iter_accuracy': per_iter_accuracies,
        'final_accuracy': per_iter_accuracies[-1] if per_iter_accuracies else 0.0
    }

    return total_loss, metrics


# =============================================================================
# Trainer Class
# =============================================================================

class OpenWebTextTrainer:
    """Training handler for OpenWebText - exactly 1 epoch."""

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

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.1),
            betas=(0.9, 0.95),
        )

        # Cosine LR scheduler
        self.total_steps = config.get('total_steps', 100000)
        warmup_steps = config.get('warmup_steps', 2000)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (self.total_steps - warmup_steps)
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
        self.tokens_seen = 0

    def train_step_accumulated(self, batch_iter, randomize_max_iters: bool = True) -> Tuple[Dict, bool]:
        """Training step with gradient accumulation. Returns (metrics, epoch_done).

        Model can early-stop when done signal > threshold AND predictions are correct,
        but is capped by a randomized max_iters to prevent dependence on any one strategy.
        """
        self.model.train()
        self.optimizer.zero_grad()

        accumulated_metrics = defaultdict(float)
        accumulated_loss = 0.0
        epoch_done = False

        for accum_idx in range(self.accumulation_steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                epoch_done = True
                break

            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            # Track tokens
            self.tokens_seen += input_ids.numel()

            # Randomize max iterations cap (model can still early-stop before this)
            if randomize_max_iters:
                max_iters = random.randint(1, self.model.max_iterations)
            else:
                max_iters = self.model.max_iterations

            if self.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    output, metadata = self.model(input_ids, target=target_ids, max_iters=max_iters)
                    loss, metrics = compute_lm_loss(
                        output, target_ids, metadata,
                        iteration_cost=self.iteration_cost,
                        done_supervision_weight=self.done_supervision_weight
                    )
                    scaled_loss = loss / self.accumulation_steps

                self.scaler.scale(scaled_loss).backward()
            else:
                output, metadata = self.model(input_ids, target=target_ids, max_iters=max_iters)
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

        if accumulated_loss > 0:
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

        num_accumulated = accum_idx + 1 if not epoch_done else accum_idx
        if num_accumulated > 0:
            final_metrics = {k: v / num_accumulated for k, v in accumulated_metrics.items()}
            final_metrics['loss'] = accumulated_loss / num_accumulated
            final_metrics['lr'] = self.scheduler.get_last_lr()[0]
        else:
            final_metrics = {}

        return final_metrics, epoch_done

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
                max_iters=self.model.max_iterations,
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

    def train_one_epoch(
        self,
        save_dir: str = 'checkpoints_openwebtext',
        log_interval: int = 100,
        eval_interval: int = 2000,
        save_interval: int = 10000
    ):
        """Train for exactly 1 epoch through OpenWebText."""
        os.makedirs(save_dir, exist_ok=True)

        actual_batch_size = self.config.get('actual_batch_size', 32)
        effective_batch_size = actual_batch_size * self.accumulation_steps
        estimated_tokens = estimate_openwebtext_tokens()

        print(f"\nStarting OpenWebText training (1 epoch)...")
        print(f"Device: {self.device}")
        print(f"Model: FlashRecursiveTransformer (125M params)")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {actual_batch_size}")
        print(f"Accumulation steps: {self.accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Max iterations: {self.model.max_iterations}")
        print(f"Estimated tokens: {estimated_tokens / 1e9:.1f}B")
        print_gpu_memory()
        print("-" * 60)

        batch_iter = iter(self.train_loader)
        running_metrics = defaultdict(float)
        running_count = 0
        start_time = time.time()
        epoch_done = False

        while not epoch_done:
            metrics, epoch_done = self.train_step_accumulated(batch_iter)

            if not metrics:
                continue

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running_metrics[k] += v
            running_count += 1

            if self.global_step % log_interval == 0 and running_count > 0:
                elapsed = time.time() - start_time
                steps_per_sec = log_interval / elapsed if elapsed > 0 else 0
                tokens_per_sec = steps_per_sec * effective_batch_size * self.config.get('max_seq_len', 512)

                avg_metrics = {k: v / running_count for k, v in running_metrics.items()}

                progress = self.tokens_seen / estimated_tokens * 100

                print(f"Step {self.global_step} | "
                      f"Progress: {progress:.2f}% | "
                      f"Loss: {avg_metrics['loss']:.4f} | "
                      f"PPL: {avg_metrics['perplexity']:.2f} | "
                      f"Avg Iters: {avg_metrics.get('avg_iterations', 0):.2f} | "
                      f"LR: {avg_metrics['lr']:.2e} | "
                      f"{tokens_per_sec/1000:.1f}K tok/s")

                for k, v in avg_metrics.items():
                    self.history[f'train_{k}'].append(v)
                self.history['step'].append(self.global_step)
                self.history['tokens_seen'].append(self.tokens_seen)

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

        # Final save
        self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'))

        print("-" * 60)
        print("Epoch complete!")
        print(f"Total steps: {self.global_step}")
        print(f"Total tokens: {self.tokens_seen:,}")
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
            'tokens_seen': self.tokens_seen,
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = defaultdict(list, checkpoint.get('history', {}))
        self.global_step = checkpoint.get('global_step', 0)
        self.tokens_seen = checkpoint.get('tokens_seen', 0)


# =============================================================================
# Main Training Function
# =============================================================================

def train_openwebtext(config: Dict = None, resume_from: str = None):
    """Main training function."""
    if config is None:
        config = {}

    print("=" * 60)
    print("125M FlashRecursiveTransformer on OpenWebText")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
        print_gpu_memory()
    else:
        print("WARNING: CUDA not available!")
        device = 'cpu'

    # Merge configs
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

    # Find optimal batch size BEFORE compilation
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
    train_loader, val_loader, tokenizer = create_openwebtext_dataloaders(
        tokenizer_name="gpt2",
        max_seq_len=full_config['max_seq_len'],
        batch_size=actual_batch_size,
        num_workers=full_config.get('num_workers', 4),
        val_samples=full_config.get('val_samples', 1000),
    )

    # Create trainer
    trainer = OpenWebTextTrainer(
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
        print(f"Resumed at step {trainer.global_step}, tokens: {trainer.tokens_seen:,}")

    history = trainer.train_one_epoch(
        save_dir=full_config.get('save_dir', 'checkpoints_openwebtext'),
        log_interval=full_config.get('log_interval', 100),
        eval_interval=full_config.get('eval_interval', 2000),
        save_interval=full_config.get('save_interval', 10000)
    )

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 125M Recursive Transformer on OpenWebText')
    parser.add_argument('--batch-size', type=int, default=32, help='Manual batch size')
    parser.add_argument('--target-batch-size', type=int, default=256, help='Target effective batch size')
    parser.add_argument('--no-auto-batch', action='store_true', help='Disable auto batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max-iters', type=int, default=5, help='Max recursive iterations')
    parser.add_argument('--seq-len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default='checkpoints_openwebtext', help='Save directory')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')

    args = parser.parse_args()

    config = {
        # Training
        'learning_rate': args.lr,
        'weight_decay': 0.1,
        'warmup_steps': 2000,
        'use_amp': True,

        # Batch size
        'batch_size': args.batch_size,
        'target_effective_batch_size': args.target_batch_size,
        'auto_batch_size': not args.no_auto_batch,
        'batch_size_safety_margin': 0.8,

        # Model
        'max_iterations': args.max_iters,
        'max_seq_len': args.seq_len,

        # Loss
        'iteration_cost': 0.01,
        'done_supervision_weight': 0.5,

        # Logging
        'save_dir': args.save_dir,
        'log_interval': 100,
        'eval_interval': 2000,
        'save_interval': 10000,
        'val_samples': 1000,
        'num_workers': args.num_workers,
    }

    train_openwebtext(config, resume_from=args.resume)
