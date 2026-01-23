"""
Training loop for Recursive Transformer on TinyStories.

Key differences from GSM8K training:
- Loss computed on ALL tokens (standard language modeling)
- Uses steps instead of epochs (streaming dataset)
- No depth tracking needed
- Goal: observe if adaptive compute emerges naturally

Features:
- Automatic batch size detection
- Gradient accumulation for target effective batch size
- GPU memory monitoring
"""

import os
import time
import random
import argparse
from typing import Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

from model import RecursiveTransformer, compute_loss
from dataset_tinystories import create_tinystories_dataloaders, get_tokenizer, TINYSTORIES_VOCAB_SIZE
from config import ExperimentConfig, ModelConfig, DataConfig, TrainConfig


# TinyStories pretraining config
TINYSTORIES_CONFIG = ExperimentConfig(
    model=ModelConfig(
        d_model=256,
        n_heads=4,
        n_layers=6,
        d_ff=1024,
        max_iterations=8,
        max_seq_len=256,
    ),
    data=DataConfig(
        data_dir='tinystories',  # Not used, we stream from HuggingFace
        batch_size=64,
        max_seq_len=256,
    ),
    train=TrainConfig(
        epochs=1,  # Not used for streaming
        learning_rate=3e-4,
        iteration_cost=0.01,
        done_supervision_weight=0.5,
    ),
    name='tinystories_pretrain'
)


# =============================================================================
# GPU Memory and Batch Size Utilities
# =============================================================================

def print_gpu_memory(prefix: str = ""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"{prefix}GPU Memory: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
    else:
        print(f"{prefix}GPU not available")


def _test_batch_size(
    model: RecursiveTransformer,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    device: str,
    gpu_mem_gb: float,
    use_amp: bool = True
) -> Tuple[bool, float]:
    """
    Test if a batch size fits in memory.

    Returns:
        (success, peak_memory_gb)
    """
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # Create dummy input
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Forward pass with max iterations (worst case memory)
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

        # Backward pass
        loss.backward()
        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        # Clean up
        model.zero_grad(set_to_none=True)
        del dummy_input, dummy_target, output, metadata, loss
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Check if we exceeded memory limit
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
    model: RecursiveTransformer,
    vocab_size: int,
    seq_len: int,
    device: str,
    start: int = 8,
    max_batch: int = 512,
    use_amp: bool = True
) -> int:
    """
    Binary search for largest batch size that fits in GPU memory.

    First doubles to find upper bound, then binary searches for optimal value.

    Args:
        model: The model to test
        vocab_size: Vocabulary size for dummy input
        seq_len: Sequence length to test
        device: Device to test on
        start: Starting batch size
        max_batch: Maximum batch size to try
        use_amp: Whether to use mixed precision (should match training)

    Returns:
        Maximum working batch size
    """
    if device != 'cuda':
        print("Auto batch size only works on CUDA, using default")
        return start

    model.train()

    # Get GPU memory limit
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Finding optimal batch size (GPU: {gpu_mem_gb:.1f}GB)...")

    # Phase 1: Double to find upper bound
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

    # If we never failed, high is beyond max_batch
    if high > max_batch:
        high = max_batch
        # Check if max_batch works
        success, peak = _test_batch_size(model, vocab_size, seq_len, high, device, gpu_mem_gb, use_amp)
        if success:
            print(f"    batch_size={high}... OK (peak: {peak:.1f}GB)")
            print(f"  Max batch size: {high} (capped at max_batch)")
            return high

    # Phase 2: Binary search between low and high
    if high > low:
        print(f"  Phase 2: Binary search between {low} and {high}...")

        while high - low > max(1, low // 8):  # Stop when within ~12% of optimal
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
        print(f"  Could not fit even batch_size={start}, using it anyway")
        return start

    print(f"  Max batch size: {low}")
    return low


def compute_batch_settings(
    model: RecursiveTransformer,
    vocab_size: int,
    seq_len: int,
    device: str,
    target_effective_batch_size: int = 128,
    auto_batch_size: bool = True,
    manual_batch_size: int = 64,
    use_amp: bool = True,
    safety_margin: float = 0.8
) -> Tuple[int, int]:
    """
    Compute optimal batch size and gradient accumulation steps.

    Args:
        model: The model
        vocab_size: Vocabulary size
        seq_len: Sequence length
        device: Device
        target_effective_batch_size: Desired effective batch size
        auto_batch_size: Whether to auto-detect batch size
        manual_batch_size: Batch size to use if auto is disabled
        use_amp: Whether using mixed precision
        safety_margin: Safety margin for batch size (0.8 = use 80% of max)

    Returns:
        (actual_batch_size, accumulation_steps)
    """
    if auto_batch_size and device == 'cuda':
        max_batch = find_max_batch_size(
            model, vocab_size, seq_len, device,
            start=8, max_batch=1024, use_amp=use_amp
        )
        actual_batch_size = max(1, int(max_batch * safety_margin))
        print(f"Max batch size: {max_batch}")
        print(f"Using batch_size={actual_batch_size} ({safety_margin*100:.0f}% safety margin)")
    else:
        actual_batch_size = manual_batch_size
        print(f"Using manual batch_size={actual_batch_size}")

    # Compute accumulation steps
    accumulation_steps = max(1, target_effective_batch_size // actual_batch_size)
    effective_batch_size = actual_batch_size * accumulation_steps

    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    return actual_batch_size, accumulation_steps


# =============================================================================
# Gradient Checkpointing
# =============================================================================

def enable_gradient_checkpointing(model: RecursiveTransformer):
    """
    Enable gradient checkpointing for the model's transformer layers.

    This saves memory by not storing intermediate activations during the forward
    pass, recomputing them during the backward pass instead. Trades compute for
    memory - typically ~30% slower but uses significantly less GPU memory.
    """
    def make_checkpointed_forward(original_forward):
        def checkpointed_forward(x, previous_states=None, attn_mask=None, return_attention=False):
            # checkpoint requires all inputs to be tensors or None
            # We need to handle previous_states (list of tensors) specially
            if previous_states is not None and len(previous_states) > 0:
                # Concatenate for checkpointing, will be handled inside the layer
                pass

            def custom_forward(x, attn_mask, return_attention_flag):
                return original_forward(x, previous_states, attn_mask, bool(return_attention_flag))

            # Use checkpoint - recomputes forward during backward
            return checkpoint(
                custom_forward,
                x, attn_mask, torch.tensor(return_attention),
                use_reentrant=False
            )
        return checkpointed_forward

    # Wrap each layer's forward method
    for layer in model.layers:
        layer._original_forward = layer.forward
        layer.forward = make_checkpointed_forward(layer._original_forward)

    model._gradient_checkpointing_enabled = True
    print("Gradient checkpointing enabled")


def disable_gradient_checkpointing(model: RecursiveTransformer):
    """Disable gradient checkpointing and restore original forward methods."""
    if not getattr(model, '_gradient_checkpointing_enabled', False):
        return

    for layer in model.layers:
        if hasattr(layer, '_original_forward'):
            layer.forward = layer._original_forward
            delattr(layer, '_original_forward')

    model._gradient_checkpointing_enabled = False
    print("Gradient checkpointing disabled")


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

    Loss on ALL tokens (standard LM), not just answer tokens.
    Done classifier learns when each position's prediction is correct.
    """
    device = output.device
    batch_size, seq_len = output.size(0), output.size(1)
    all_outputs = metadata['all_outputs']
    done_logits = metadata['done_logits']  # [batch, num_iters, seq_len]
    done_probs = metadata['done_probs']    # [batch, num_iters, seq_len]
    num_iters = len(all_outputs)

    # Ignore index for padding
    ignore_idx = -100

    # 1. Task loss on final output (all tokens)
    task_loss = F.cross_entropy(
        output.view(-1, output.size(-1)),
        target.view(-1),
        ignore_index=ignore_idx,
        reduction='mean'
    )

    # 2. Per-position done classifier supervision
    # For each position at each iteration: is the prediction correct?
    per_position_correct = []
    per_iter_accuracies = []

    for iter_idx, iter_output in enumerate(all_outputs):
        predictions = iter_output.argmax(dim=-1)  # [batch, seq_len]
        correct = (predictions == target).float()  # [batch, seq_len]

        # Mask out padding for accuracy computation
        valid_mask = (target != ignore_idx)
        if valid_mask.any():
            acc = correct[valid_mask].mean().item()
        else:
            acc = 0.0

        per_position_correct.append(correct)
        per_iter_accuracies.append(acc)

    # Stack: [batch, num_iters, seq_len]
    done_targets = torch.stack(per_position_correct, dim=1)

    # Cumulative max over iterations: once correct, stays "done"
    done_targets_cummax, _ = torch.cummax(done_targets, dim=1)

    # Create mask for valid positions (non-padding)
    valid_mask = (target != ignore_idx).unsqueeze(1).expand(-1, num_iters, -1)

    # BCE loss only on valid positions
    done_supervision_loss = F.binary_cross_entropy_with_logits(
        done_logits,
        done_targets_cummax,
        reduction='none'
    )
    done_supervision_loss = (done_supervision_loss * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

    # 3. Per-position iteration cost
    # Penalize positions for "continuing" (not being done)
    continuation_probs = 1 - done_probs  # [batch, num_iters, seq_len]
    valid_mask_single = (target != ignore_idx)  # [batch, seq_len]
    expected_iters_per_pos = continuation_probs.sum(dim=1)  # [batch, seq_len]

    if valid_mask_single.any():
        iter_loss = iteration_cost * expected_iters_per_pos[valid_mask_single].mean()
    else:
        iter_loss = torch.tensor(0.0, device=device)

    # Total loss
    total_loss = task_loss + done_supervision_weight * done_supervision_loss + iter_loss

    # Compute perplexity
    with torch.no_grad():
        perplexity = torch.exp(task_loss).item()

        # Average iterations per position
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

class TinyStoriesTrainer:
    """Training handler for TinyStories language modeling with gradient accumulation."""

    def __init__(
        self,
        model: RecursiveTransformer,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda',
        accumulation_steps: int = 1,
        gradient_checkpointing: bool = False
    ):
        self.model = model.to(device)

        # Enable gradient checkpointing if requested
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            enable_gradient_checkpointing(self.model)
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
        self.use_amp = config.get('use_amp', False) and device == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Loss weights
        self.iteration_cost = config.get('iteration_cost', 0.01)
        self.done_supervision_weight = config.get('done_supervision_weight', 0.5)

        # Tracking
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)
        self.global_step = 0
        self.micro_step = 0  # Tracks individual batches for accumulation

    def train_step_accumulated(
        self,
        train_iter,
        random_max_iters: bool = True
    ) -> Tuple[Dict, any]:
        """
        Training step with gradient accumulation.

        Returns:
            metrics: Accumulated metrics for this step
            train_iter: Updated iterator (in case it was reset)
        """
        self.model.train()
        self.optimizer.zero_grad()

        accumulated_metrics = defaultdict(float)
        accumulated_loss = 0.0

        for accum_idx in range(self.accumulation_steps):
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            # Randomize max_iterations per batch (key for learning adaptive compute)
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
                    # Scale loss for accumulation
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

            # Accumulate metrics
            accumulated_loss += loss.item()
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    accumulated_metrics[k] += v

            self.micro_step += 1

        # Optimizer step after accumulation
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

        # Average accumulated metrics
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

            # Run full iterations for evaluation
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

        # Average
        total_loss /= num_batches
        for k in total_metrics:
            total_metrics[k] /= num_batches

        return {'loss': total_loss, **total_metrics}

    def train(
        self,
        total_steps: int,
        save_dir: str = 'checkpoints_tinystories',
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000
    ):
        """Main training loop using steps instead of epochs."""
        os.makedirs(save_dir, exist_ok=True)

        actual_batch_size = self.config.get('actual_batch_size', self.config.get('batch_size', 64))
        effective_batch_size = actual_batch_size * self.accumulation_steps

        print(f"\nStarting TinyStories training for {total_steps} steps...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {actual_batch_size}")
        print(f"Accumulation steps: {self.accumulation_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Max iterations: {self.model.max_iterations}")
        print(f"Gradient checkpointing: {self.gradient_checkpointing}")
        print_gpu_memory()
        print("-" * 60)

        train_iter = iter(self.train_loader)
        running_metrics = defaultdict(float)
        running_count = 0
        start_time = time.time()

        while self.global_step < total_steps:
            # Training step with accumulation
            metrics, train_iter = self.train_step_accumulated(train_iter)

            # Accumulate for logging
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running_metrics[k] += v
            running_count += 1

            # Logging
            if self.global_step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = log_interval / elapsed if elapsed > 0 else 0
                samples_per_sec = steps_per_sec * effective_batch_size

                # Average metrics
                avg_metrics = {k: v / running_count for k, v in running_metrics.items()}

                print(f"Step {self.global_step}/{total_steps} | "
                      f"Loss: {avg_metrics['loss']:.4f} | "
                      f"PPL: {avg_metrics['perplexity']:.2f} | "
                      f"Avg Iters: {avg_metrics['avg_iterations']:.2f} | "
                      f"LR: {avg_metrics['lr']:.2e} | "
                      f"{steps_per_sec:.1f} steps/s ({samples_per_sec:.0f} samples/s)")

                # Record history
                for k, v in avg_metrics.items():
                    self.history[f'train_{k}'].append(v)
                self.history['step'].append(self.global_step)

                # Reset
                running_metrics = defaultdict(float)
                running_count = 0
                start_time = time.time()

            # Validation
            if self.global_step % eval_interval == 0:
                val_metrics = self.validate()
                print(f"  [Val] Loss: {val_metrics['loss']:.4f} | "
                      f"PPL: {val_metrics['perplexity']:.2f} | "
                      f"Avg Iters: {val_metrics.get('avg_iterations', 0):.2f}")

                # Record
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        self.history[f'val_{k}'].append(v)

                # Print gate values and memory
                gates = self.model.get_gate_values()
                print(f"  Cross-attn gates: {[f'{g:.3f}' for g in gates]}")
                print_gpu_memory("  ")

                # Save best
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'))
                    print(f"  New best model saved!")

                print()

            # Regular save
            if self.global_step % save_interval == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_{self.global_step}.pt'))

        # Final save
        self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'))

        print("-" * 60)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print_gpu_memory("Final ")

        return dict(self.history)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history),
            'global_step': self.global_step,
            'accumulation_steps': self.accumulation_steps,
            'gradient_checkpointing': self.gradient_checkpointing
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns global step to resume from."""
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

def train_tinystories(config: Dict = None, resume_from: str = None):
    """
    Main training function for TinyStories.

    Args:
        config: Training configuration dict (uses TINYSTORIES_CONFIG if None)
        resume_from: Path to checkpoint to resume from
    """
    if config is None:
        config = TINYSTORIES_CONFIG.to_dict()

    # Device
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
        print_gpu_memory()
    else:
        print("WARNING: CUDA not available!")
        device = 'cpu'

    # Get tokenizer first for vocab size (GPT-Neo like TinyStories paper)
    tokenizer = get_tokenizer(config.get('tokenizer_name', 'EleutherAI/gpt-neo-125M'))

    # Create model (use 10K vocab like TinyStories paper)
    model = RecursiveTransformer(
        vocab_size=config.get('vocab_size', TINYSTORIES_VOCAB_SIZE),
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        max_iterations=config.get('max_iterations', 8),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 256)
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Compute batch size and accumulation
    model_on_device = model.to(device)
    actual_batch_size, accumulation_steps = compute_batch_settings(
        model=model_on_device,
        vocab_size=config.get('vocab_size', TINYSTORIES_VOCAB_SIZE),
        seq_len=config.get('max_seq_len', 256),
        device=device,
        target_effective_batch_size=config.get('target_effective_batch_size', 128),
        auto_batch_size=config.get('auto_batch_size', True),
        manual_batch_size=config.get('batch_size', 64),
        use_amp=config.get('use_amp', True),
        safety_margin=config.get('batch_size_safety_margin', 0.8)
    )

    # Update config with actual values
    config['actual_batch_size'] = actual_batch_size
    config['accumulation_steps'] = accumulation_steps

    # Create dataloaders with actual batch size (GPT-Neo tokenizer, 10K vocab)
    train_loader, val_loader, tokenizer = create_tinystories_dataloaders(
        tokenizer_name=config.get('tokenizer_name', 'EleutherAI/gpt-neo-125M'),
        max_seq_len=config.get('max_seq_len', 256),
        batch_size=actual_batch_size,
        num_workers=0,  # Must be 0 for streaming
        val_samples=config.get('val_samples', 1000)
    )

    # Create trainer
    trainer = TinyStoriesTrainer(
        model=model_on_device,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        accumulation_steps=accumulation_steps,
        gradient_checkpointing=config.get('gradient_checkpointing', False)
    )

    # Resume if needed
    if resume_from is not None:
        print(f"Resuming from {resume_from}...")
        trainer.load_checkpoint(resume_from)
        print(f"Resumed at step {trainer.global_step}")

    # Train
    history = trainer.train(
        total_steps=config.get('total_steps', 50000),
        save_dir=config.get('save_dir', 'checkpoints_tinystories'),
        log_interval=config.get('log_interval', 100),
        eval_interval=config.get('eval_interval', 1000),
        save_interval=config.get('save_interval', 5000)
    )

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Recursive Transformer on TinyStories')
    parser.add_argument('--steps', type=int, default=50000, help='Total training steps')
    parser.add_argument('--batch-size', type=int, default=64, help='Manual batch size (if auto disabled)')
    parser.add_argument('--target-batch-size', type=int, default=128, help='Target effective batch size')
    parser.add_argument('--no-auto-batch', action='store_true', help='Disable auto batch size detection')
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension (512 gives ~40M params)')
    parser.add_argument('--n-layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--max-iters', type=int, default=6, help='Max recursive iterations')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--save-dir', type=str, default='checkpoints_tinystories', help='Save directory')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory (slower but uses less VRAM)')

    args = parser.parse_args()

    # Compute n_heads ensuring d_model is divisible
    n_heads = 8 if args.d_model % 8 == 0 else (6 if args.d_model % 6 == 0 else 4)

    config = {
        'vocab_size': TINYSTORIES_VOCAB_SIZE,  # 10K (like TinyStories paper)
        'd_model': args.d_model,
        'n_heads': n_heads,
        'n_layers': args.n_layers,
        'd_ff': args.d_model * 4,
        'max_iterations': args.max_iters,
        'dropout': 0.1,
        'max_seq_len': 256,

        'batch_size': args.batch_size,
        'target_effective_batch_size': args.target_batch_size,
        'auto_batch_size': not args.no_auto_batch,
        'batch_size_safety_margin': 0.8,

        'total_steps': args.steps,
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'iteration_cost': 0.01,
        'done_supervision_weight': 0.5,
        'use_amp': True,
        'gradient_checkpointing': args.gradient_checkpointing,

        'save_dir': args.save_dir,
        'log_interval': 100,
        'eval_interval': 1000,
        'save_interval': 5000,
        'val_samples': 1000,
    }

    train_tinystories(config, resume_from=args.resume)
