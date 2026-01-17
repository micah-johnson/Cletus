"""
Training loop for Recursive Transformer.
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
from dataset import create_dataloaders, ArithmeticTokenizer


class Trainer:
    """Training and evaluation handler for RecursiveTransformer."""

    def __init__(
        self,
        model: RecursiveTransformer,
        tokenizer: ArithmeticTokenizer,
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
        # None = use model's default max_iterations (no curriculum)
        self.curriculum = config.get('curriculum', None)
        self.current_max_iters = None  # Set during training

    def get_curriculum_max_iters(self, epoch: int) -> int:
        """Get max_iterations for current epoch based on curriculum schedule."""
        if self.curriculum is None:
            return self.model.max_iterations

        # Curriculum format: [(epoch_threshold, max_iters), ...]
        # e.g., [(25, 2), (50, 3), (75, 4), (None, 6)]
        for threshold, max_iters in self.curriculum:
            if threshold is None or epoch < threshold:
                return max_iters
        return self.model.max_iterations

    def train_epoch(self, max_iters: int = None, random_max_iters: bool = True) -> Dict:
        """Run one training epoch.

        Args:
            max_iters: If set, limit iterations via force_iterations (for curriculum)
            random_max_iters: If True, randomize max_iterations per batch (1 to max_iters)
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = defaultdict(float)
        num_batches = 0
        total_iterations = 0

        # Determine effective max_iters
        effective_max_iters = max_iters if max_iters is not None else self.model.max_iterations

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            # Randomize max_iterations per batch to prevent iteration-dependent strategies
            if random_max_iters:
                batch_max_iters = random.randint(1, effective_max_iters)
            else:
                batch_max_iters = effective_max_iters

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    output, metadata = self.model(input_ids, force_iterations=batch_max_iters)
                    loss, metrics = compute_loss(
                        output, target_ids, metadata,
                        iteration_cost=self.iteration_cost,
                        done_supervision_weight=self.done_supervision_weight,
                        pad_token_id=self.tokenizer.pad_token_id
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
                    pad_token_id=self.tokenizer.pad_token_id
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            epoch_loss += loss.item()
            total_iterations += metadata['num_iterations']
            for k, v in metrics.items():
                # Skip list metrics (like per_iter_accuracy)
                if isinstance(v, (int, float)):
                    epoch_metrics[k] += v
            num_batches += 1

        # Average metrics
        epoch_loss /= num_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        # Add average iterations during training
        epoch_metrics['avg_iterations'] = total_iterations / num_batches

        return {'loss': epoch_loss, **epoch_metrics}

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None, max_iters: int = None) -> Dict:
        """Run validation.

        Args:
            loader: DataLoader to use (defaults to val_loader)
            max_iters: If set, cap iterations (for curriculum) but allow early stopping
        """
        self.model.eval()
        loader = loader or self.val_loader

        total_loss = 0.0
        total_metrics = defaultdict(float)
        num_batches = 0

        # Track iterations and accuracy by depth
        depth_iterations = defaultdict(list)
        depth_correct = defaultdict(int)
        depth_total = defaultdict(int)
        correct_predictions = 0
        total_predictions = 0

        # Temporarily cap model's max_iterations for curriculum (allows early stopping)
        original_max_iters = self.model.max_iterations
        if max_iters is not None:
            self.model.max_iterations = max_iters

        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            depths = batch['depth']

            # Use threshold-based stopping (no force_iterations)
            output, metadata = self.model(
                input_ids,
                threshold=self.config.get('done_threshold', 0.7)
            )
            loss, metrics = compute_loss(
                output, target_ids, metadata,
                iteration_cost=self.iteration_cost,
                done_supervision_weight=self.done_supervision_weight,
                pad_token_id=self.tokenizer.pad_token_id
            )

            total_loss += loss.item()
            for k, v in metrics.items():
                # Skip list metrics (like per_iter_accuracy)
                if isinstance(v, (int, float)):
                    total_metrics[k] += v
            num_batches += 1

            # Track iterations per depth
            for i, d in enumerate(depths.tolist()):
                depth_iterations[d].append(metadata['num_iterations'])

            # Check prediction accuracy per sample and track by depth
            predictions = output.argmax(dim=-1)
            for i in range(len(input_ids)):
                pred_tokens = predictions[i].tolist()
                target_tokens = target_ids[i].tolist()
                d = depths[i].item()

                # Check if all non-pad tokens match
                matches = sum(1 for p, t in zip(pred_tokens, target_tokens)
                             if t != self.tokenizer.pad_token_id and p == t)
                total_valid = sum(1 for t in target_tokens if t != self.tokenizer.pad_token_id)

                is_correct = (matches == total_valid)
                if is_correct:
                    correct_predictions += 1
                    depth_correct[d] += 1
                depth_total[d] += 1
                total_predictions += 1

        # Average metrics
        total_loss /= num_batches
        for k in total_metrics:
            total_metrics[k] /= num_batches

        # Compute depth stats (iterations + accuracy)
        depth_stats = {}
        for depth in sorted(set(depth_iterations.keys()) | set(depth_total.keys())):
            iters = depth_iterations.get(depth, [])
            correct = depth_correct.get(depth, 0)
            total = depth_total.get(depth, 1)
            depth_stats[depth] = {
                'mean_iterations': sum(iters) / len(iters) if iters else 0,
                'accuracy': correct / total if total > 0 else 0,
                'count': total
            }

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Restore original max_iterations
        self.model.max_iterations = original_max_iters

        return {
            'loss': total_loss,
            'accuracy': accuracy,
            'depth_stats': depth_stats,
            **total_metrics
        }

    def train(
        self,
        epochs: int,
        save_dir: str = 'checkpoints',
        log_interval: int = 10
    ) -> Dict:
        """Full training loop."""
        os.makedirs(save_dir, exist_ok=True)

        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.curriculum:
            print(f"Curriculum: {self.curriculum}")
        print("-" * 60)

        for epoch in range(epochs):
            start_time = time.time()

            # Get curriculum max_iters for this epoch
            max_iters = self.get_curriculum_max_iters(epoch)
            self.current_max_iters = max_iters

            # Train with curriculum-limited iterations
            train_metrics = self.train_epoch(max_iters=max_iters)

            # Validate with same iteration limit
            val_metrics = self.validate(max_iters=max_iters)

            # Step scheduler
            self.scheduler.step()

            # Record history
            for k, v in train_metrics.items():
                self.history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                if k != 'depth_stats':
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
                print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                      f"Acc: {val_metrics['accuracy']:.2%} | "
                      f"Done prob: {val_metrics.get('mean_done_prob', 0):.3f}")

                # Print depth -> iterations and accuracy
                print("  Depth -> Iters | Accuracy:")
                for depth, stats in sorted(val_metrics['depth_stats'].items()):
                    print(f"    {depth}: {stats['mean_iterations']:.2f} iters | {stats['accuracy']:.1%} ({stats['count']} samples)")

                # Print gate values
                gates = self.model.get_gate_values()
                print(f"  Cross-attn gates: {[f'{g:.3f}' for g in gates]}")
                print()

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'))

        # Save final model
        self.save_checkpoint(os.path.join(save_dir, 'final_model.pt'))

        print("-" * 60)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return dict(self.history)

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'history': dict(self.history)
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = defaultdict(list, checkpoint.get('history', {}))
        return checkpoint.get('config', {})


def train_model(config: Dict) -> Tuple[RecursiveTransformer, Dict]:
    """
    Main training function.

    Args:
        config: Training configuration dict

    Returns:
        Trained model and training history
    """
    # Device detection with diagnostics
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("WARNING: CUDA not available! Training on CPU (slow)")
        print("To fix: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        device = 'cpu'
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        train_samples=config.get('train_samples', 10000),
        val_samples=config.get('val_samples', 1000),
        test_samples=config.get('test_samples', 1000),
        max_depth=config.get('max_depth', 4),
        max_seq_len=config.get('max_seq_len', 64),
        batch_size=config.get('batch_size', 32)
    )

    # Create model
    model = RecursiveTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 3),
        d_ff=config.get('d_ff', 256),
        max_iterations=config.get('max_iterations', 6),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 64)
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    history = trainer.train(
        epochs=config.get('epochs', 50),
        save_dir=config.get('save_dir', 'checkpoints'),
        log_interval=config.get('log_interval', 5)
    )

    # Final test evaluation
    print("\nFinal Test Evaluation:")
    test_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    print("Depth -> Iters | Accuracy:")
    for depth, stats in sorted(test_metrics['depth_stats'].items()):
        print(f"  {depth}: {stats['mean_iterations']:.2f} iters | {stats['accuracy']:.1%}")

    return model, history


if __name__ == '__main__':
    # Default configuration
    config = {
        # Model
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'd_ff': 256,
        'max_iterations': 6,
        'dropout': 0.1,

        # Data
        'train_samples': 10000,
        'val_samples': 1000,
        'test_samples': 1000,
        'max_depth': 4,
        'max_seq_len': 64,
        'batch_size': 32,

        # Training
        'epochs': 50,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'min_lr': 1e-6,
        'iteration_cost': 0.01,
        'done_supervision_weight': 0.5,
        'done_threshold': 0.7,
        'use_amp': False,

        # Misc
        'save_dir': 'checkpoints',
        'log_interval': 5
    }

    model, history = train_model(config)
