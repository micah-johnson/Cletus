"""
Interactive demo for the Recursive Transformer.

Usage:
    python demo.py checkpoints/best_model.pt
    python demo.py checkpoints/best_model.pt --max-tokens 100 --temperature 0.8
"""

import argparse
import torch
from typing import Optional, Union

from model import RecursiveTransformer, FlashRecursiveTransformer
from dataset_tinystories import get_tokenizer, TINYSTORIES_VOCAB_SIZE


def load_model(checkpoint_path: str, device: str = 'cuda', force_flash: bool = None):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        force_flash: If True, use FlashRecursiveTransformer. If False, use RecursiveTransformer.
                    If None, auto-detect from checkpoint config.
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    # Determine model type
    if force_flash is not None:
        use_flash = force_flash
    else:
        use_flash = config.get('use_flash_attention', False)

    ModelClass = FlashRecursiveTransformer if use_flash else RecursiveTransformer
    model_type = "FlashRecursiveTransformer" if use_flash else "RecursiveTransformer"

    model = ModelClass(
        vocab_size=config.get('vocab_size', TINYSTORIES_VOCAB_SIZE),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 8),
        d_ff=config.get('d_ff', 2048),
        max_iterations=config.get('max_iterations', 6),
        dropout=0.0,
        max_seq_len=config.get('max_seq_len', 256)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model type: {model_type}")
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  d_model={config.get('d_model')}, n_layers={config.get('n_layers')}, "
          f"max_iterations={config.get('max_iterations')}")

    return model, config


@torch.no_grad()
def generate(
    model: Union[RecursiveTransformer, FlashRecursiveTransformer],
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    threshold: float = 0.5,
    show_iterations: bool = True,
    device: str = 'cuda'
) -> str:
    """Generate text from a prompt."""
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], device=device)

    # Generate
    output_ids, iterations_per_token = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        threshold=threshold,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode
    generated_ids = output_ids[0, len(tokens):].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if show_iterations and iterations_per_token:
        avg_iters = sum(iterations_per_token) / len(iterations_per_token)
        print(f"\n  [Iterations per token: avg={avg_iters:.1f}, "
              f"min={min(iterations_per_token)}, max={max(iterations_per_token)}]")

    return generated_text


def interactive_demo(
    model: RecursiveTransformer,
    tokenizer,
    device: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    threshold: float = 0.5
):
    """Run interactive demo loop."""
    print("\n" + "=" * 60)
    print("RECURSIVE TRANSFORMER DEMO")
    print("=" * 60)
    print(f"Settings: max_tokens={max_tokens}, temperature={temperature}, "
          f"top_k={top_k}, threshold={threshold}")
    print("\nCommands:")
    print("  Type a prompt and press Enter to generate")
    print("  /temp <value>  - Change temperature (e.g., /temp 0.8)")
    print("  /tokens <n>    - Change max tokens (e.g., /tokens 100)")
    print("  /topk <n>      - Change top-k (e.g., /topk 40)")
    print("  /threshold <v> - Change done threshold (e.g., /threshold 0.7)")
    print("  /settings      - Show current settings")
    print("  /quit or /q    - Exit")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        # Handle commands
        if prompt.startswith('/'):
            parts = prompt.split()
            cmd = parts[0].lower()

            if cmd in ['/quit', '/q', '/exit']:
                print("Goodbye!")
                break
            elif cmd == '/temp' and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature value")
            elif cmd == '/tokens' and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"Max tokens set to {max_tokens}")
                except ValueError:
                    print("Invalid token count")
            elif cmd == '/topk' and len(parts) > 1:
                try:
                    top_k = int(parts[1])
                    print(f"Top-k set to {top_k}")
                except ValueError:
                    print("Invalid top-k value")
            elif cmd == '/threshold' and len(parts) > 1:
                try:
                    threshold = float(parts[1])
                    print(f"Threshold set to {threshold}")
                except ValueError:
                    print("Invalid threshold value")
            elif cmd == '/settings':
                print(f"Current settings:")
                print(f"  temperature: {temperature}")
                print(f"  max_tokens: {max_tokens}")
                print(f"  top_k: {top_k}")
                print(f"  threshold: {threshold}")
            else:
                print(f"Unknown command: {cmd}")
            continue

        # Generate
        print()
        generated = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            threshold=threshold,
            show_iterations=True,
            device=device
        )

        print(f"\n{prompt}{generated}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Interactive demo for Recursive Transformer')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--max-tokens', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--threshold', type=float, default=0.5, help='Done classifier threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt (non-interactive mode)')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model and tokenizer
    model, config = load_model(args.checkpoint, args.device)
    tokenizer = get_tokenizer()

    if args.prompt:
        # Single generation mode
        generated = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            threshold=args.threshold,
            show_iterations=True,
            device=args.device
        )
        print(f"\n{args.prompt}{generated}\n")
    else:
        # Interactive mode
        interactive_demo(
            model, tokenizer, args.device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            threshold=args.threshold
        )


if __name__ == '__main__':
    main()
