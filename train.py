"""
Training loop for Recursive Transformer on GSM8K.
"""

import os
import time
import random
from typing import Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler

from model import RecursiveTransformer, compute_loss
from dataset import create_gsm8k_dataloaders, get_tokenizer


class Trainer:
    """Training and evaluation handler for RecursiveTransformer."""

    def __init__(
        self,
        model: RecursiveTransformer,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 3e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 50),
            eta_min=config.get('min_lr', 1e-6)
        )

        # Mixed precision training
        self.use_amp = config.get('use_amp', False) and device == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Loss weights
        self.iteration_cost = config.get('iteration_cost', 0.01)
        self.done_supervision_weight = config.get('done_supervision_weight', 0.5)

        # Tracking
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)

        # Curriculum: max_iterations schedule by epoch
        self.curriculum = config.get('curriculum', None)
        self.current_max_iters = None

    def get_curriculum_max_iters(self, epoch: int) -> int:
        """Get max_iterations for current epoch based on curriculum schedule."""
        if self.curriculum is None:
            return self.model.max_iterations

        for threshold, max_iters in self.curriculum:
            if threshold is None or epoch < threshold:
                return max_iters
        return self.model.max_iterations

    def train_epoch(self, max_iters: int = None, random_max_iters: bool = True) -> Dict:
        """Run one training epoch."""
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        num_batches = 0

        effective_max_iters = max_iters if max_iters is not None else self.model.max_iterations

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            # Randomize max_iterations per batch
            if random_max_iters:
                batch_max_iters = random.randint(1, effective_max_iters)
            else:
                batch_max_iters = effective_max_iters

            self.optimizer.zero_grad()

            # Use -100 as ignore index (standard for HuggingFace)
            pad_token_id = -100

            if self.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    output, metadata = self.model(input_ids, force_iterations=batch_max_iters)
                    loss, metrics = compute_loss(
                        output, target_ids, metadata,
                        iteration_cost=self.iteration_cost,
                        done_supervision_weight=self.done_supervision_weight,
                        pad_token_id=pad_token_id
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output, metadata = self.model(input_ids, force_iterations=batch_max_iters)
                loss, metrics = compute_loss(
                    output, target_ids, metadata,
                    iteration_cost=self.iteration_cost,
                    done_supervision_weight=self.done_supervision_weight,
                    pad_token_id=pad_token_id
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    epoch_metrics[k] += v
            num_batches += 1

        # Average metrics
        epoch_loss /= num_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        return {'loss': epoch_loss, **epoch_metrics}

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None, max_iters: int = None) -> Dict:
        """Run validation."""
        self.model.eval()
        loader = loader or self.val_loader

        total_loss = 0.0
        total_metrics = defaultdict(float)
        num_batches = 0

        correct_predictions = 0
        total_predictions = 0
        total_iterations_used = 0

        # Per-depth tracking
        depth_correct = defaultdict(int)
        depth_total = defaultdict(int)
        depth_iterations = defaultdict(list)

        original_max_iters = self.model.max_iterations
        if max_iters is not None:
            self.model.max_iterations = max_iters

        pad_token_id = -100

        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            depths = batch.get('depth', None)

            output, metadata = self.model(
                input_ids,
                threshold=self.config.get('done_threshold', 0.7)
            )
            loss, metrics = compute_loss(
                output, target_ids, metadata,
                iteration_cost=self.iteration_cost,
                done_supervision_weight=self.done_supervision_weight,
                pad_token_id=pad_token_id,
                tokenizer=self.tokenizer  # For number-only accuracy
            )

            total_loss += loss.item()

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    total_metrics[k] += v
            num_batches += 1

            # Check prediction accuracy per sample
            predictions = output.argmax(dim=-1)
            mask = (target_ids != pad_token_id)
            sample_correct = ((predictions == target_ids) | ~mask).all(dim=-1)

            # Per-position iteration tracking
            iters_per_pos = metadata.get('iterations_per_position', None)

            for i in range(input_ids.size(0)):
                is_correct = sample_correct[i].item()
                correct_predictions += int(is_correct)
                total_predictions += 1

                # Compute avg iterations for this sample (over answer positions)
                if iters_per_pos is not None:
                    sample_mask = mask[i]
                    if sample_mask.any():
                        sample_avg_iters = iters_per_pos[i][sample_mask].mean().item()
                    else:
                        sample_avg_iters = metadata['num_iterations']
                else:
                    sample_avg_iters = metadata['num_iterations']

                total_iterations_used += sample_avg_iters

                # Track per-depth stats if depth is available
                if depths is not None:
                    d = depths[i].item() if hasattr(depths[i], 'item') else depths[i]
                    depth_total[d] += 1
                    depth_correct[d] += int(is_correct)
                    depth_iterations[d].append(sample_avg_iters)

        # Average metrics
        total_loss /= num_batches
        for k in total_metrics:
            total_metrics[k] /= num_batches

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_iterations = total_iterations_used / total_predictions if total_predictions > 0 else 0.0

        # Compute per-depth stats
        depth_stats = {}
        for d in sorted(set(depth_total.keys())):
            depth_stats[d] = {
                'accuracy': depth_correct[d] / depth_total[d] if depth_total[d] > 0 else 0.0,
                'avg_iters': sum(depth_iterations[d]) / len(depth_iterations[d]) if depth_iterations[d] else 0.0,
                'count': depth_total[d]
            }

        self.model.max_iterations = original_max_iters

        return {
            'loss': total_loss,
            'accuracy': accuracy,
            'avg_iterations': avg_iterations,
            'depth_stats': depth_stats,
            **total_metrics
        }

    def train(
        self,
        epochs: int,
        save_dir: str = 'checkpoints',
        log_interval: int = 10,
        start_epoch: int = 0
    ) -> Dict:
        """Full training loop."""
        os.makedirs(save_dir, exist_ok=True)

        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch + 1}...")
        else:
            print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.curriculum:
            print(f"Curriculum: {self.curriculum}")
        print("-" * 60)

        for epoch in range(start_epoch, epochs):
            start_time = time.time()

            max_iters = self.get_curriculum_max_iters(epoch)
            self.current_max_iters = max_iters

            train_metrics = self.train_epoch(max_iters=max_iters)
            val_metrics = self.validate(max_iters=max_iters)

            self.scheduler.step()

            # Record history (skip non-scalar values like depth_stats)
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    self.history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    self.history[f'val_{k}'].append(v)

            epoch_time = time.time() - start_time

            # Print progress
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                iter_info = f" [max_iters={max_iters}]" if self.curriculum else ""
                print(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s){iter_info}")
                print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                      f"Task: {train_metrics['task_loss']:.4f} | "
                      f"Acc: {train_metrics.get('final_accuracy', 0):.2%} | "
                      f"Avg Iters: {train_metrics.get('avg_iterations', 0):.2f}")
                num_acc = val_metrics.get('number_accuracy')
                num_acc_str = f" | NumAcc: {num_acc:.2%}" if num_acc is not None else ""
                print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                      f"Acc: {val_metrics['accuracy']:.2%}{num_acc_str} | "
                      f"Avg Iters: {val_metrics.get('avg_iterations', 0):.2f}")

                # Print per-depth stats if available
                depth_stats = val_metrics.get('depth_stats', {})
                if depth_stats:
                    print("  Per-depth: ", end="")
                    parts = []
                    for d in sorted(depth_stats.keys()):
                        s = depth_stats[d]
                        parts.append(f"D{d}: {s['accuracy']:.0%} ({s['avg_iters']:.1f} iters)")
                    print(" | ".join(parts))

                # Print gate values
                gates = self.model.get_gate_values()
                print(f"  Cross-attn gates: {[f'{g:.3f}' for g in gates]}")
                print()

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'), epoch=epoch)

        # Save final model
        self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'), epoch=epoch)

        print("-" * 60)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return dict(self.history)

    def save_checkpoint(self, path: str, epoch: int = 0):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history),
            'epoch': epoch
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint. Returns the epoch to resume from."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = defaultdict(list, checkpoint.get('history', {}))
        return checkpoint.get('epoch', 0)


def train_model(config: Dict, resume_from: str = None) -> Tuple[RecursiveTransformer, Dict]:
    """
    Main training function.

    Args:
        config: Training configuration dict
        resume_from: Path to checkpoint to resume from (optional)

    Returns:
        Trained model and training history
    """
    # Device detection
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("WARNING: CUDA not available! Training on CPU (slow)")
        device = 'cpu'
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, test_loader, tokenizer = create_gsm8k_dataloaders(
        data_dir=config.get('data_dir', 'data/gsm8k'),
        tokenizer_name=config.get('tokenizer_name', 'gpt2'),
        max_seq_len=config.get('max_seq_len', 512),
        batch_size=config.get('batch_size', 16),
        num_workers=config.get('num_workers', 4),
        simple_mode=config.get('simple_mode', True)
    )

    # Create model
    model = RecursiveTransformer(
        vocab_size=config.get('vocab_size', tokenizer.vocab_size),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 2048),
        max_iterations=config.get('max_iterations', 8),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 512)
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer (use test_loader as val_loader for now)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=test_loader,  # Use test as validation
        config=config,
        device=device
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from is not None:
        print(f"Loading checkpoint from {resume_from}...")
        start_epoch = trainer.load_checkpoint(resume_from) + 1  # Start from next epoch
        print(f"Resumed from epoch {start_epoch}")

    # Train
    history = trainer.train(
        epochs=config.get('epochs', 50),
        save_dir=config.get('save_dir', 'checkpoints'),
        log_interval=config.get('log_interval', 10),
        start_epoch=start_epoch
    )

    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Avg Iterations: {test_metrics.get('avg_iterations', 0):.2f}")

    return model, history


if __name__ == '__main__':
    # Default configuration
    config = {
        # Model
        'vocab_size': 50257,  # GPT-2 vocab size
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_iterations': 8,
        'dropout': 0.1,

        # Data
        'data_dir': 'data/gsm8k',
        'tokenizer_name': 'gpt2',
        'max_seq_len': 512,
        'batch_size': 16,
        'num_workers': 4,
        'simple_mode': True,

        # Training
        'epochs': 30,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'min_lr': 1e-6,
        'iteration_cost': 0.01,
        'done_supervision_weight': 0.5,
        'done_threshold': 0.7,
        'use_amp': True,

        # Misc
        'save_dir': 'checkpoints',
        'log_interval': 5
    }

    model, history = train_model(config)
