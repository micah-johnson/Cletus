"""
Recursive Transformer Experiment - Main Entry Point

A transformer that can "think again" by attending to its previous hidden states,
with a learned classifier that decides when it's done thinking.

Usage:
    python main.py train --config small
    python main.py train --config tiny --epochs 10
    python main.py evaluate --checkpoint checkpoints/best_model.pt
    python main.py demo --checkpoint checkpoints/best_model.pt

"""

import argparse
import os
import random
import sys

import numpy as np
import torch

from config import get_config, ExperimentConfig, DEFAULT_CURRICULUM
from model import RecursiveTransformer
from dataset import create_dataloaders, ArithmeticTokenizer, generate_nested_arithmetic
from train import Trainer, train_model
from evaluate import RecursiveTransformerAnalyzer, run_full_evaluation


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cmd_train(args):
    """Train the model."""
    # Load config
    config = get_config(args.config)

    # Override with command line args
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.learning_rate = args.lr
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.max_iterations:
        config.model.max_iterations = args.max_iterations
    if args.iteration_cost:
        config.train.iteration_cost = args.iteration_cost
    if args.curriculum:
        config.train.curriculum = DEFAULT_CURRICULUM

    # Set seed
    set_seed(config.seed)

    print("=" * 60)
    print("RECURSIVE TRANSFORMER EXPERIMENT")
    print("=" * 60)
    print(f"Config: {config.name}")
    print(f"Model: d={config.model.d_model}, h={config.model.n_heads}, "
          f"L={config.model.n_layers}, max_iter={config.model.max_iterations}")
    print(f"Data: {config.data.train_samples} train, depth={config.data.max_depth}")
    print(f"Training: {config.train.epochs} epochs, lr={config.train.learning_rate}")
    if config.train.curriculum:
        print(f"Curriculum: {config.train.curriculum}")
    print("=" * 60 + "\n")

    # Train
    model, history = train_model(config.to_dict())

    # Save config
    config.save(os.path.join(config.train.save_dir, 'config.json'))

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {config.train.save_dir}/")

    return model, history


def cmd_evaluate(args):
    """Evaluate a trained model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint.get('config', {})
    tokenizer = ArithmeticTokenizer()

    # Recreate model
    model = RecursiveTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 3),
        d_ff=config.get('d_ff', 256),
        max_iterations=config.get('max_iterations', 6),
        dropout=0.0,  # No dropout for evaluation
        max_seq_len=config.get('max_seq_len', 64)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded. Running evaluation...")

    # Run evaluation
    results = run_full_evaluation(
        model=model,
        tokenizer=tokenizer,
        test_samples=args.test_samples,
        max_depth=config.get('max_depth', 4),
        output_dir=args.output_dir,
        device=device,
        max_seq_len=config.get('max_seq_len', 64)
    )

    # Plot training history if available
    if 'history' in checkpoint:
        from evaluate import RecursiveTransformerAnalyzer
        analyzer = RecursiveTransformerAnalyzer(model, tokenizer, device)
        analyzer.visualize_training_history(
            checkpoint['history'],
            save_path=os.path.join(args.output_dir, 'training_history.png')
        )

    return results


def cmd_demo(args):
    """Interactive demo with a trained model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint.get('config', {})
    tokenizer = ArithmeticTokenizer()

    # Recreate model
    model = RecursiveTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get('d_model', 128),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 3),
        d_ff=config.get('d_ff', 256),
        max_iterations=config.get('max_iterations', 6),
        dropout=0.0,
        max_seq_len=config.get('max_seq_len', 64)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    analyzer = RecursiveTransformerAnalyzer(model, tokenizer, device)
    max_seq_len = config.get('max_seq_len', 64)
    done_threshold = config.get('done_threshold', 0.7)

    print(checkpoint.get('config', {}))

    print("\n" + "=" * 60)
    print("RECURSIVE TRANSFORMER DEMO")
    print("=" * 60)
    print("Enter arithmetic expressions or commands:")
    print("  'random N' - Generate random expression of depth N")
    print("  'quit' - Exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Expression > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        if user_input.lower().startswith('random'):
            try:
                depth = int(user_input.split()[1])
            except (IndexError, ValueError):
                depth = 3
            sample = generate_nested_arithmetic(max_depth=6, target_depth=depth)
            expression = sample['expression']
            expected = sample['result']
            print(f"Generated: {expression}")
        else:
            expression = user_input
            try:
                expected = eval(expression)
            except:
                expected = "?"

        # Analyze with threshold-based early stopping
        analysis = analyzer.analyze_single_sample(
            expression, expected,
            max_seq_len=max_seq_len,
            threshold=done_threshold
        )

        # Check correctness
        try:
            is_correct = (int(analysis['predicted_result']) == expected)
            status = "CORRECT" if is_correct else "WRONG"
        except:
            status = "WRONG"

        print(f"\nExpression: {expression}")
        print(f"Expected:   {expected}")
        print(f"Predicted:  {analysis['predicted_result']} [{status}]")
        print(f"Iterations: {analysis['num_iterations']}")
        print(f"Done probs: {[f'{p:.2f}' for p in analysis['done_probs'].flatten()]}")
        print()


def cmd_sample(args):
    """Show sample expressions from the dataset."""
    print("\nSample Expressions by Depth:")
    print("-" * 60)

    for depth in range(1, args.max_depth + 1):
        print(f"\nDepth {depth}:")
        for i in range(3):
            sample = generate_nested_arithmetic(max_depth=args.max_depth, target_depth=depth)
            result = sample['result']
            expr = sample['expression']
            print(f"  {expr} = {result}")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Recursive Transformer Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='small',
                              choices=['tiny', 'small', 'medium', 'large'],
                              help='Configuration preset')
    train_parser.add_argument('--epochs', type=int, help='Override number of epochs')
    train_parser.add_argument('--lr', type=float, help='Override learning rate')
    train_parser.add_argument('--batch-size', type=int, help='Override batch size')
    train_parser.add_argument('--max-iterations', type=int, help='Override max iterations')
    train_parser.add_argument('--iteration-cost', type=float, help='Override iteration cost')
    train_parser.add_argument('--curriculum', action='store_true',
                              help='Enable curriculum learning (gradually unlock iterations)')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')
    eval_parser.add_argument('--test-samples', type=int, default=500,
                             help='Number of test samples')
    eval_parser.add_argument('--output-dir', type=str, default='results',
                             help='Output directory for results')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Interactive demo')
    demo_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')

    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Show sample expressions')
    sample_parser.add_argument('--max-depth', type=int, default=4,
                               help='Maximum expression depth')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'demo':
        cmd_demo(args)
    elif args.command == 'sample':
        cmd_sample(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
