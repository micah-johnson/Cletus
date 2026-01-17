"""
Training loop for baseline transformers (weight-tied and standard 12M).
"""

import os
import time
import argparse
import random
from typing import Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.amp import GradScaler

from model import StandardWeightTiedTransformer, StandardTransformer, compute_loss_standard
from dataset import create_dataloaders, ArithmeticTokenizer
from config import get_baseline_config, get_standard_12m_config


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BaselineTrainer:
    """Training handler for StandardWeightTiedTransformer."""

    def __init__(
        self,
        model: StandardWeightTiedTransformer,
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

        # Mixed precision
        self.use_amp = config.get('use_amp', False) and device == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Tracking
        self.best_val_loss = float('inf')
        self.history = defaultdict(list)

    def train_epoch(self) -> Dict:
        """Run one training epoch."""
        self.model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch in self.train_loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast('cuda', dtype=torch.bfloat16):
                    output, _ = self.model(input_ids)
                    loss, metrics = compute_loss_standard(
                        output, target_ids,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output, _ = self.model(input_ids)
                loss, metrics = compute_loss_standard(
                    output, target_ids,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += metrics['accuracy']
            num_batches += 1

        return {
            'loss': epoch_loss / num_batches,
            'accuracy': epoch_acc / num_batches
        }

    @torch.no_grad()
    def validate(self, loader=None) -> Dict:
        """Run validation."""
        self.model.eval()
        loader = loader or self.val_loader

        total_loss = 0.0
        num_batches = 0

        # Track accuracy by depth
        depth_correct = defaultdict(int)
        depth_total = defaultdict(int)

        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            depths = batch['depth']

            output, _ = self.model(input_ids)
            loss, _ = compute_loss_standard(
                output, target_ids,
                pad_token_id=self.tokenizer.pad_token_id
            )

            total_loss += loss.item()
            num_batches += 1

            # Per-sample accuracy by depth
            predictions = output.argmax(dim=-1)
            for i in range(len(input_ids)):
                pred_tokens = predictions[i].tolist()
                target_tokens = target_ids[i].tolist()
                d = depths[i].item()

                matches = sum(1 for p, t in zip(pred_tokens, target_tokens)
                             if t != self.tokenizer.pad_token_id and p == t)
                total_valid = sum(1 for t in target_tokens if t != self.tokenizer.pad_token_id)

                if matches == total_valid:
                    depth_correct[d] += 1
                depth_total[d] += 1

        # Compute stats
        depth_stats = {}
        total_correct = 0
        total_samples = 0
        for depth in sorted(depth_total.keys()):
            correct = depth_correct[depth]
            total = depth_total[depth]
            depth_stats[depth] = {
                'accuracy': correct / total if total > 0 else 0,
                'count': total
            }
            total_correct += correct
            total_samples += total

        return {
            'loss': total_loss / num_batches,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0,
            'depth_stats': depth_stats
        }

    def train(self, epochs: int, save_dir: str = 'checkpoints_baseline', log_interval: int = 10):
        """Full training loop."""
        os.makedirs(save_dir, exist_ok=True)

        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if hasattr(self.model, 'effective_depth'):
            print(f"Effective depth: {self.model.effective_depth} layers")
        else:
            print(f"Layers: {self.model.n_layers}")
        print("-" * 60)

        for epoch in range(epochs):
            start_time = time.time()

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.scheduler.step()

            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            epoch_time = time.time() - start_time

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2%}")
                print(f"  Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2%}")
                print("  Depth -> Accuracy:")
                for depth, stats in sorted(val_metrics['depth_stats'].items()):
                    print(f"    {depth}: {stats['accuracy']:.1%} ({stats['count']} samples)")
                print()

            # Save best
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'))

        # Save final
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


def train_baseline(config: Dict, model_type: str = 'weight_tied'):
    """Main training function for baseline models.

    Args:
        config: Training configuration dict
        model_type: 'weight_tied' for StandardWeightTiedTransformer,
                   'standard_12m' for StandardTransformer (12M params)
    """
    # Device
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        device = 'cpu'

    # Create dataloaders
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        train_samples=config.get('train_samples', 50000),
        val_samples=config.get('val_samples', 5000),
        test_samples=config.get('test_samples', 5000),
        max_depth=config.get('max_depth', 5),
        max_seq_len=config.get('max_seq_len', 192),
        batch_size=config.get('batch_size', 256)
    )

    # Create model based on type
    if model_type == 'standard_12m':
        model = StandardTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config.get('d_model', 288),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 12),
            d_ff=config.get('d_ff', 1152),
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('max_seq_len', 192)
        )
        print(f"\nModel: StandardTransformer (12M)")
        print(f"  Layers: {config.get('n_layers', 12)} (NOT weight-tied)")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        model = StandardWeightTiedTransformer(
            vocab_size=tokenizer.vocab_size,
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 4),
            d_ff=config.get('d_ff', 1050),
            n_repeats=config.get('n_repeats', 3),
            dropout=config.get('dropout', 0.1),
            max_seq_len=config.get('max_seq_len', 192)
        )
        print(f"\nModel: StandardWeightTiedTransformer")
        print(f"  Unique layers: {config.get('n_layers', 4)}")
        print(f"  Repeats: {config.get('n_repeats', 3)}")
        print(f"  Effective depth: {config.get('n_layers', 4) * config.get('n_repeats', 3)}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = BaselineTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )

    # Train
    history = trainer.train(
        epochs=config.get('epochs', 100),
        save_dir=config.get('save_dir', 'checkpoints_baseline'),
        log_interval=config.get('log_interval', 5)
    )

    # Final test
    print("\nFinal Test Evaluation:")
    test_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2%}")
    print("Depth -> Accuracy:")
    for depth, stats in sorted(test_metrics['depth_stats'].items()):
        print(f"  {depth}: {stats['accuracy']:.1%}")

    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train baseline transformers')
    parser.add_argument('--model', type=str, default='weight_tied',
                        choices=['weight_tied', 'standard_12m'],
                        help='Model type: weight_tied (3M) or standard_12m (12M)')
    parser.add_argument('--epochs', type=int, help='Override epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--n-repeats', type=int, help='Override n_repeats (weight_tied only)')

    args = parser.parse_args()

    # Load config based on model type
    if args.model == 'standard_12m':
        config = get_standard_12m_config()
        model_type = 'standard_12m'
    else:
        config = get_baseline_config('medium')
        model_type = 'weight_tied'

    # Overrides
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.learning_rate = args.lr
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.n_repeats and model_type == 'weight_tied':
        config.model.n_repeats = args.n_repeats

    set_seed(config.seed)

    print("=" * 60)
    if model_type == 'standard_12m':
        print("PRACTICAL COMPARISON: Standard Transformer (12M)")
        print("=" * 60)
        print(f"Model: d={config.model.d_model}, h={config.model.n_heads}, L={config.model.n_layers}")
    else:
        print("BASELINE: Standard Weight-Tied Transformer (3M)")
        print("=" * 60)
        print(f"Model: d={config.model.d_model}, h={config.model.n_heads}, "
              f"L={config.model.n_layers}x{config.model.n_repeats}")
    print(f"Data: {config.data.train_samples} train, depth={config.data.max_depth}")
    print(f"Training: {config.train.epochs} epochs, lr={config.train.learning_rate}")
    print("=" * 60 + "\n")

    train_baseline(config.to_dict(), model_type=model_type)


if __name__ == '__main__':
    main()
