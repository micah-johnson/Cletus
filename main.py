"""
Recursive Transformer Experiment - Main Entry Point

A transformer that can "think again" by attending to its previous hidden states,
with a learned classifier that decides when it's done thinking.

Now configured for GSM8K math word problems with Llama tokenizer.

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
from dataset import create_gsm8k_dataloaders, get_tokenizer
from train import Trainer, train_model


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
    print("RECURSIVE TRANSFORMER - GSM8K")
    print("=" * 60)
    print(f"Config: {config.name}")
    print(f"Model: d={config.model.d_model}, h={config.model.n_heads}, "
          f"L={config.model.n_layers}, max_iter={config.model.max_iterations}")
    print(f"Data: {config.data.data_dir}, batch_size={config.data.batch_size}")
    print(f"Training: {config.train.epochs} epochs, lr={config.train.learning_rate}")
    if config.train.curriculum:
        print(f"Curriculum: {config.train.curriculum}")
    if args.resume:
        print(f"Resuming from: {args.resume}")
    print("=" * 60 + "\n")

    # Train
    model, history = train_model(config.to_dict(), resume_from=args.resume)

    # Save config
    os.makedirs(config.train.save_dir, exist_ok=True)
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
    tokenizer = get_tokenizer(config.get('tokenizer_name', 'gpt2'))

    # Recreate model
    model = RecursiveTransformer(
        vocab_size=config.get('vocab_size', tokenizer.vocab_size),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 2048),
        max_iterations=config.get('max_iterations', 8),
        dropout=0.0,  # No dropout for evaluation
        max_seq_len=config.get('max_seq_len', 512)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load test data
    _, test_loader, _ = create_gsm8k_dataloaders(
        data_dir=config.get('data_dir', 'data/gsm8k'),
        tokenizer_name=config.get('tokenizer_name', 'gpt2'),
        max_seq_len=config.get('max_seq_len', 512),
        batch_size=args.batch_size,
        num_workers=0
    )

    # Evaluate
    correct = 0
    total = 0
    total_iterations = 0

    print("\nEvaluating...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            output, metadata = model(
                input_ids,
                threshold=config.get('done_threshold', 0.7)
            )

            # Check predictions (simplified - just check if output matches target)
            predictions = output.argmax(dim=-1)
            mask = (target_ids != -100)
            batch_correct = ((predictions == target_ids) | ~mask).all(dim=-1).sum().item()

            correct += batch_correct
            total += input_ids.size(0)
            total_iterations += metadata['num_iterations'] * input_ids.size(0)

    accuracy = correct / total if total > 0 else 0
    avg_iterations = total_iterations / total if total > 0 else 0

    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"  Avg iterations: {avg_iterations:.2f}")

    return {'accuracy': accuracy, 'avg_iterations': avg_iterations}


def cmd_demo(args):
    """Interactive demo with a trained model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint.get('config', {})
    tokenizer = get_tokenizer(config.get('tokenizer_name', 'gpt2'))

    # Recreate model
    model = RecursiveTransformer(
        vocab_size=config.get('vocab_size', tokenizer.vocab_size),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 2048),
        max_iterations=config.get('max_iterations', 8),
        dropout=0.0,
        max_seq_len=config.get('max_seq_len', 512)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    max_seq_len = config.get('max_seq_len', 512)
    done_threshold = config.get('done_threshold', 0.7)

    print("\n" + "=" * 60)
    print("RECURSIVE TRANSFORMER DEMO - GSM8K")
    print("=" * 60)
    print("Enter a math word problem (or 'quit' to exit):")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Tokenize question as prompt
        prompt_tokens = tokenizer.encode(user_input, add_special_tokens=True, max_length=max_seq_len - 32, truncation=True)
        prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        # Generate autoregressively with per-token iteration tracking
        output_ids, iterations_per_token = model.generate(
            prompt_ids,
            max_new_tokens=32,
            threshold=done_threshold,
            temperature=0.7,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        # Decode generated tokens (excluding prompt)
        generated_tokens = output_ids[0, len(prompt_tokens):].tolist()
        # Stop at EOS or PAD
        answer_tokens = []
        for tok in generated_tokens:
            if tok == tokenizer.eos_token_id or tok == tokenizer.pad_token_id:
                break
            answer_tokens.append(tok)

        predicted_answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        print(f"\nQuestion: {user_input}")
        print(f"Predicted answer: {predicted_answer}")
        print(f"Tokens generated: {len(answer_tokens)}")
        print(f"Iterations per token: {iterations_per_token[:len(answer_tokens)]}")
        print(f"Avg iterations: {sum(iterations_per_token[:len(answer_tokens)]) / max(len(answer_tokens), 1):.2f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Recursive Transformer Experiment - GSM8K',
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
    train_parser.add_argument('--resume', type=str, default=None,
                              help='Path to checkpoint to resume training from')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')
    eval_parser.add_argument('--batch-size', type=int, default=16,
                             help='Batch size for evaluation')

    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Interactive demo')
    demo_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'demo':
        cmd_demo(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
