"""
Analyze cross-attention patterns across iterations in the Recursive Transformer.

This script helps understand whether the model primarily attends to the immediately
previous iteration or distributes attention across all past iterations.

Usage:
    python analyze_cross_attention.py checkpoint.pt --prompt "Once upon a time"
    python analyze_cross_attention.py checkpoint.pt --num-samples 100
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from model import RecursiveTransformer
from dataset_tinystories import get_tokenizer, TINYSTORIES_VOCAB_SIZE


def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[RecursiveTransformer, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    model = RecursiveTransformer(
        vocab_size=config.get('vocab_size', TINYSTORIES_VOCAB_SIZE),
        d_model=config.get('d_model', 512),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 8),
        d_ff=config.get('d_ff', 2048),
        max_iterations=config.get('max_iterations', 6),
        dropout=0.0,  # No dropout for inference
        max_seq_len=config.get('max_seq_len', 256)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


def extract_cross_attention_by_iteration(
    cross_attention_weights: List[List[torch.Tensor]],
    seq_len: int
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Extract cross-attention weights organized by iteration.

    Args:
        cross_attention_weights: List[iteration][layer] of attention weight tensors
            Each tensor has shape [batch, seq_len, num_prev_iters * seq_len] (averaged over heads)
            or [batch, num_heads, seq_len, num_prev_iters * seq_len]
        seq_len: Sequence length

    Returns:
        Dict mapping iteration -> Dict mapping prev_iteration -> attention weights
        Attention weights are averaged over batch, heads, and query positions
    """
    results = {}

    for iter_idx, layer_attns in enumerate(cross_attention_weights):
        if not layer_attns:
            continue

        # Current iteration attends to iterations 0..iter_idx-1
        num_prev_iters = iter_idx + 1  # At iteration 1, we have 1 prev state (iter 0)

        iter_results = {}

        for layer_idx, attn in enumerate(layer_attns):
            if attn is None:
                continue

            # Handle different attention weight shapes
            # PyTorch MultiheadAttention returns [batch, seq_len, seq_len] when average_attn_weights=True (default)
            # or [batch, num_heads, seq_len, seq_len] when average_attn_weights=False
            if attn.dim() == 3:
                # Shape: [batch, q_len, kv_len] - already averaged over heads
                batch_size, q_len, kv_len = attn.shape
                # Add head dim for consistent processing
                attn = attn.unsqueeze(1)  # [batch, 1, q_len, kv_len]

            batch_size, num_heads, q_len, kv_len = attn.shape

            # Reshape to separate previous iterations
            # [batch, heads, q_len, num_prev, seq_len]
            attn_by_iter = attn.view(batch_size, num_heads, q_len, num_prev_iters, seq_len)

            # Sum attention over key positions within each previous iteration
            # [batch, heads, q_len, num_prev]
            attn_per_prev_iter = attn_by_iter.sum(dim=-1)

            # Average over batch, heads, and query positions
            # [num_prev]
            avg_attn = attn_per_prev_iter.mean(dim=(0, 1, 2))

            for prev_iter in range(num_prev_iters):
                if prev_iter not in iter_results:
                    iter_results[prev_iter] = []
                iter_results[prev_iter].append(avg_attn[prev_iter].item())

        # Average across layers
        for prev_iter in iter_results:
            iter_results[prev_iter] = np.mean(iter_results[prev_iter])

        results[iter_idx + 1] = iter_results  # +1 because iteration 0 has no cross-attn

    return results


def analyze_attention_patterns(
    model: RecursiveTransformer,
    tokenizer,
    prompts: List[str],
    device: str = 'cuda',
    force_iterations: Optional[int] = None
) -> Dict:
    """
    Analyze cross-attention patterns across multiple prompts.

    Returns detailed statistics about which previous iterations are attended to.
    """
    all_iter_attention = defaultdict(lambda: defaultdict(list))
    all_gate_values = []

    for prompt in prompts:
        # Tokenize
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([tokens], device=device)

        # Run forward with attention weights
        with torch.no_grad():
            output, metadata = model(
                input_ids,
                return_all_states=True,
                force_iterations=force_iterations or model.max_iterations
            )

        cross_attn = metadata.get('cross_attention_weights', [])
        if not cross_attn:
            continue

        seq_len = input_ids.size(1)

        # Extract attention by iteration
        iter_attn = extract_cross_attention_by_iteration(cross_attn, seq_len)

        for curr_iter, prev_attns in iter_attn.items():
            for prev_iter, attn_weight in prev_attns.items():
                all_iter_attention[curr_iter][prev_iter].append(attn_weight)

        # Get gate values
        all_gate_values.append(model.get_gate_values())

    # Compute statistics
    stats = {
        'iteration_attention': {},
        'relative_attention': {},
        'gate_values': np.mean(all_gate_values, axis=0).tolist() if all_gate_values else []
    }

    for curr_iter in sorted(all_iter_attention.keys()):
        prev_attns = all_iter_attention[curr_iter]
        iter_stats = {}
        total_attn = 0

        for prev_iter in sorted(prev_attns.keys()):
            mean_attn = np.mean(prev_attns[prev_iter])
            iter_stats[prev_iter] = mean_attn
            total_attn += mean_attn

        stats['iteration_attention'][curr_iter] = iter_stats

        # Compute relative attention (normalized)
        if total_attn > 0:
            stats['relative_attention'][curr_iter] = {
                prev: attn / total_attn for prev, attn in iter_stats.items()
            }

    return stats


def print_attention_analysis(stats: Dict):
    """Print formatted analysis of attention patterns."""
    print("\n" + "=" * 70)
    print("CROSS-ATTENTION ANALYSIS")
    print("=" * 70)

    # Gate values
    if stats['gate_values']:
        print("\nCross-Attention Gate Values (per layer):")
        for i, gate in enumerate(stats['gate_values']):
            bar = "█" * int(gate * 20)
            print(f"  Layer {i}: {gate:.3f} {bar}")

    # Iteration attention patterns
    print("\n" + "-" * 70)
    print("ATTENTION TO PREVIOUS ITERATIONS")
    print("-" * 70)
    print("\nFor each current iteration, shows how much attention goes to each previous iteration.")
    print("Higher values = more attention to that previous iteration.\n")

    iter_attn = stats['iteration_attention']
    rel_attn = stats['relative_attention']

    if not iter_attn:
        print("No cross-attention data available.")
        return

    # Find max iterations for formatting
    max_iter = max(iter_attn.keys())

    # Print header
    header = "Curr Iter │ " + " │ ".join([f"Prev {i}" for i in range(max_iter)])
    print(header)
    print("─" * len(header))

    for curr_iter in sorted(iter_attn.keys()):
        prev_attns = iter_attn[curr_iter]
        rel = rel_attn.get(curr_iter, {})

        # Raw attention values
        values = []
        for prev_iter in range(curr_iter):
            if prev_iter in prev_attns:
                val = prev_attns[prev_iter]
                rel_val = rel.get(prev_iter, 0)
                values.append(f"{val:.3f} ({rel_val*100:4.1f}%)")
            else:
                values.append("   -   ")

        print(f"    {curr_iter}     │ " + " │ ".join(values))

    # Summary statistics
    print("\n" + "-" * 70)
    print("SUMMARY: Attention to Specific Iterations")
    print("-" * 70)

    # Compute attention to iteration 0 vs others (excluding iter 1 which only has iter 0)
    iter0_attention = []
    immediate_prev_attention = []
    middle_attention = []

    for curr_iter, prev_attns in rel_attn.items():
        if curr_iter < 2:
            continue  # Skip iter 1 - it only has iter 0

        for prev_iter, attn in prev_attns.items():
            if prev_iter == 0:
                iter0_attention.append(attn)
            elif prev_iter == curr_iter - 1:
                immediate_prev_attention.append(attn)
            else:
                middle_attention.append(attn)

    print("\nAttention distribution (from iterations 2+):\n")

    if iter0_attention:
        mean_iter0 = np.mean(iter0_attention) * 100
        bar = "█" * int(mean_iter0 / 2)
        print(f"  {'Iteration 0 (first)':>25}: {mean_iter0:5.1f}% {bar}")

    if middle_attention:
        mean_middle = np.mean(middle_attention) * 100
        bar = "█" * int(mean_middle / 2)
        print(f"  {'Middle iterations':>25}: {mean_middle:5.1f}% {bar}")

    if immediate_prev_attention:
        mean_prev = np.mean(immediate_prev_attention) * 100
        bar = "█" * int(mean_prev / 2)
        print(f"  {'Immediately previous':>25}: {mean_prev:5.1f}% {bar}")

    # Key insight
    print("\n" + "-" * 70)
    print("KEY INSIGHT:")

    if iter0_attention and immediate_prev_attention:
        mean_iter0 = np.mean(iter0_attention) * 100
        mean_prev = np.mean(immediate_prev_attention) * 100

        if mean_iter0 > 70:
            print(f"  The model primarily attends to ITERATION 0 ({mean_iter0:.1f}%)")
            print(f"  The immediately previous iteration only gets {mean_prev:.1f}%")
            print("  → The model 'looks back to the beginning' rather than chaining thoughts.")
            print("  → Iteration 0 contains the original token embeddings + first pass processing.")
        elif mean_prev > 50:
            print(f"  The model primarily attends to the IMMEDIATELY PREVIOUS iteration ({mean_prev:.1f}%)")
            print(f"  Iteration 0 gets {mean_iter0:.1f}%")
            print("  → This suggests a 'chain of thought' pattern.")
        else:
            print(f"  Attention is distributed: iter 0 gets {mean_iter0:.1f}%, previous gets {mean_prev:.1f}%")
            print("  → The model uses a mix of initial and recent information.")


def visualize_attention_heatmap(stats: Dict, save_path: Optional[str] = None):
    """Create a heatmap visualization of attention patterns."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping visualization")
        return

    rel_attn = stats['relative_attention']
    if not rel_attn:
        return

    max_iter = max(rel_attn.keys())

    # Create matrix
    matrix = np.zeros((max_iter, max_iter))
    for curr_iter, prev_attns in rel_attn.items():
        for prev_iter, attn in prev_attns.items():
            matrix[curr_iter - 1, prev_iter] = attn * 100

    # Mask upper triangle (future iterations)
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        xticklabels=[f'Iter {i}' for i in range(max_iter)],
        yticklabels=[f'Iter {i+1}' for i in range(max_iter)],
        ax=ax,
        cbar_kws={'label': 'Attention %'}
    )
    ax.set_xlabel('Previous Iteration (attended to)')
    ax.set_ylabel('Current Iteration (attending from)')
    ax.set_title('Cross-Attention Distribution Across Iterations')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nHeatmap saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze cross-attention patterns')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to analyze')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of random prompts to analyze')
    parser.add_argument('--force-iters', type=int, default=None,
                        help='Force specific number of iterations')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Save heatmap to file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)
    print(f"Model loaded: d_model={config.get('d_model')}, "
          f"n_layers={config.get('n_layers')}, max_iters={config.get('max_iterations')}")

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Generate prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        # Use variety of prompts
        base_prompts = [
            "Once upon a time",
            "The little girl",
            "One day, a",
            "There was a",
            "The boy and his",
            "A magic",
            "In the forest",
            "The princess",
            "A small dog",
            "The sun was",
        ]
        # Repeat to get enough samples
        prompts = (base_prompts * (args.num_samples // len(base_prompts) + 1))[:args.num_samples]

    print(f"\nAnalyzing {len(prompts)} prompts...")

    # Analyze
    stats = analyze_attention_patterns(
        model, tokenizer, prompts,
        device=args.device,
        force_iterations=args.force_iters
    )

    # Print results
    print_attention_analysis(stats)

    # Visualize
    if args.save_plot or not args.prompt:
        visualize_attention_heatmap(stats, args.save_plot)


if __name__ == '__main__':
    main()
