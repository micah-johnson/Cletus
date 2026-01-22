"""
Analysis script for iteration patterns in TinyStories-trained recursive transformer.

Analyzes:
- Iterations per token during generation
- Correlation between iterations and token frequency (rare vs common)
- Correlation between iterations and position in sentence
- Correlation between iterations and surprisal/perplexity
- PPL at different iteration counts (to show adaptive compute helps)
"""

import argparse
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import RecursiveTransformer
from dataset_tinystories import get_tokenizer, get_token_frequencies, TinyStoriesDatasetFinite, TINYSTORIES_VOCAB_SIZE
from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[RecursiveTransformer, Dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = RecursiveTransformer(
        vocab_size=config.get('vocab_size', TINYSTORIES_VOCAB_SIZE),
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 4),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1024),
        max_iterations=config.get('max_iterations', 8),
        dropout=0.0,  # No dropout during inference
        max_seq_len=config.get('max_seq_len', 256)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, config


@torch.no_grad()
def analyze_generation(
    model: RecursiveTransformer,
    tokenizer,
    prompt: str,
    max_tokens: int = 50,
    threshold: float = 0.5,
    temperature: float = 0.5,
    top_k: Optional[int] = 50,
    token_frequencies: Optional[Dict[int, int]] = None
) -> List[Dict]:
    """
    Generate text and analyze iteration count per token.

    Returns list of dicts with:
        - token: the generated token string
        - token_id: the token ID
        - iterations: number of iterations used
        - done_prob: final done probability
        - frequency: token frequency in corpus (if provided)
        - log_prob: log probability of the token
        - position: position in generated sequence
    """
    device = next(model.parameters()).device

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    current_ids = input_ids.clone()

    results = []
    prompt_len = input_ids.size(1)

    for pos in range(max_tokens):
        batch_size, seq_len = current_ids.shape

        # Forward pass through all iterations
        output, metadata = model(
            current_ids,
            threshold=threshold,
            force_iterations=None  # Let done classifier decide
        )

        # Get iterations used for the last position
        iters_per_pos = metadata['iterations_per_position']  # [batch, seq_len]
        iterations_used = int(iters_per_pos[0, -1].item())

        # Get done probability at last position
        done_probs = metadata['done_probs']  # [batch, num_iters, seq_len]
        final_done_prob = done_probs[0, -1, -1].item()

        # Get logits for next token
        logits = output[0, -1, :]  # [vocab_size]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        token_str = tokenizer.decode([token_id])
        log_prob = log_probs[token_id].item()

        # Get frequency if available
        frequency = None
        freq_category = "unknown"
        if token_frequencies is not None:
            frequency = token_frequencies.get(token_id, 0)
            total = sum(token_frequencies.values())
            if frequency == 0:
                freq_category = "unseen"
            elif frequency < total * 0.0001:
                freq_category = "rare"
            elif frequency < total * 0.001:
                freq_category = "uncommon"
            elif frequency < total * 0.01:
                freq_category = "common"
            else:
                freq_category = "very_common"

        results.append({
            'token': token_str,
            'token_id': token_id,
            'iterations': iterations_used,
            'done_prob': final_done_prob,
            'frequency': frequency,
            'freq_category': freq_category,
            'log_prob': log_prob,
            'surprisal': -log_prob / math.log(2),  # In bits
            'position': pos
        })

        # Check for EOS
        if token_id == tokenizer.eos_token_id:
            break

        # Append and continue
        current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

    return results


def print_generation_analysis(results: List[Dict], prompt: str = ""):
    """Pretty print generation analysis."""
    print("\n" + "=" * 70)
    print(f"Prompt: \"{prompt}\"")
    print("=" * 70)
    print(f"{'Token':<15} {'Iters':>5} {'DoneP':>6} {'Surp':>6} {'Freq':>12}")
    print("-" * 70)

    for r in results:
        token_display = repr(r['token'])[1:-1]  # Remove quotes
        if len(token_display) > 12:
            token_display = token_display[:12] + "..."

        print(f"{token_display:<15} {r['iterations']:>5} {r['done_prob']:>6.2f} "
              f"{r['surprisal']:>6.1f} {r['freq_category']:>12}")

    print("-" * 70)

    # Summary statistics
    avg_iters = sum(r['iterations'] for r in results) / len(results)
    avg_surprisal = sum(r['surprisal'] for r in results) / len(results)

    print(f"Average iterations: {avg_iters:.2f}")
    print(f"Average surprisal: {avg_surprisal:.2f} bits")
    print()


def analyze_iteration_correlations(results: List[Dict]) -> Dict:
    """Analyze correlations between iterations and other factors."""
    if not results:
        return {}

    # Group by frequency category
    by_freq = defaultdict(list)
    for r in results:
        by_freq[r['freq_category']].append(r['iterations'])

    freq_avg_iters = {cat: sum(iters) / len(iters)
                      for cat, iters in by_freq.items() if iters}

    # Correlation with surprisal
    if len(results) > 1:
        iters = [r['iterations'] for r in results]
        surprisals = [r['surprisal'] for r in results]

        # Simple Pearson correlation
        n = len(results)
        mean_i = sum(iters) / n
        mean_s = sum(surprisals) / n

        cov = sum((i - mean_i) * (s - mean_s) for i, s in zip(iters, surprisals)) / n
        std_i = math.sqrt(sum((i - mean_i) ** 2 for i in iters) / n)
        std_s = math.sqrt(sum((s - mean_s) ** 2 for s in surprisals) / n)

        if std_i > 0 and std_s > 0:
            correlation_surprisal = cov / (std_i * std_s)
        else:
            correlation_surprisal = 0.0
    else:
        correlation_surprisal = 0.0

    # Correlation with position
    if len(results) > 1:
        positions = [r['position'] for r in results]
        iters = [r['iterations'] for r in results]

        n = len(results)
        mean_p = sum(positions) / n
        mean_i = sum(iters) / n

        cov = sum((p - mean_p) * (i - mean_i) for p, i in zip(positions, iters)) / n
        std_p = math.sqrt(sum((p - mean_p) ** 2 for p in positions) / n)
        std_i = math.sqrt(sum((i - mean_i) ** 2 for i in iters) / n)

        if std_p > 0 and std_i > 0:
            correlation_position = cov / (std_p * std_i)
        else:
            correlation_position = 0.0
    else:
        correlation_position = 0.0

    return {
        'avg_iters_by_frequency': freq_avg_iters,
        'correlation_with_surprisal': correlation_surprisal,
        'correlation_with_position': correlation_position
    }


def run_comprehensive_analysis(
    model: RecursiveTransformer,
    tokenizer,
    token_frequencies: Optional[Dict[int, int]] = None,
    num_prompts: int = 10
):
    """Run comprehensive analysis across multiple prompts."""

    prompts = [
        "Once upon a time",
        "The little girl",
        "One day, a",
        "There was a",
        "A big dog",
        "The sun was",
        "In the forest",
        "The boy wanted to",
        "She looked at the",
        "They went to the"
    ][:num_prompts]

    all_results = []

    print("\n" + "=" * 70)
    print("COMPREHENSIVE ITERATION ANALYSIS")
    print("=" * 70)

    for prompt in prompts:
        results = analyze_generation(
            model, tokenizer, prompt,
            max_tokens=30,
            token_frequencies=token_frequencies
        )
        all_results.extend(results)
        print_generation_analysis(results, prompt)

    # Overall analysis
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)

    correlations = analyze_iteration_correlations(all_results)

    print("\nAverage iterations by token frequency:")
    for cat in ['very_common', 'common', 'uncommon', 'rare', 'unseen', 'unknown']:
        if cat in correlations['avg_iters_by_frequency']:
            print(f"  {cat:>12}: {correlations['avg_iters_by_frequency'][cat]:.2f}")

    print(f"\nCorrelation with surprisal: {correlations['correlation_with_surprisal']:.3f}")
    print(f"Correlation with position: {correlations['correlation_with_position']:.3f}")

    # Hypothesis check
    print("\n" + "-" * 70)
    print("HYPOTHESIS CHECK: Do harder tokens use more iterations?")
    print("-" * 70)

    corr = correlations['correlation_with_surprisal']
    if corr > 0.3:
        print(f"Strong positive correlation ({corr:.3f}) - HYPOTHESIS SUPPORTED")
        print("Higher surprisal (harder to predict) tokens use more iterations!")
    elif corr > 0.1:
        print(f"Weak positive correlation ({corr:.3f}) - Partial support")
        print("Some tendency for harder tokens to use more iterations.")
    elif corr > -0.1:
        print(f"No significant correlation ({corr:.3f}) - Inconclusive")
        print("Iterations don't strongly correlate with token difficulty.")
    else:
        print(f"Negative correlation ({corr:.3f}) - HYPOTHESIS NOT SUPPORTED")
        print("Easier tokens actually use more iterations (unexpected).")

    return all_results, correlations


def analyze_specific_examples(
    model: RecursiveTransformer,
    tokenizer,
    token_frequencies: Optional[Dict[int, int]] = None
):
    """Analyze specific interesting examples."""

    examples = [
        # Easy continuation expected
        ("Once upon a", "time"),  # Very predictable
        ("The dog ran", "away"),  # Predictable
        ("She said", "hello"),    # Common

        # Hard continuation expected
        ("The wizard's name was", "Merlin"),  # Specific name
        ("The capital of", "France"),         # Factual
        ("The year was", "1984"),              # Specific number
    ]

    print("\n" + "=" * 70)
    print("SPECIFIC EXAMPLE ANALYSIS")
    print("=" * 70)

    for prompt, expected in examples:
        results = analyze_generation(
            model, tokenizer, prompt,
            max_tokens=5,
            token_frequencies=token_frequencies
        )

        if results:
            first_token = results[0]
            print(f"\nPrompt: \"{prompt}\"")
            print(f"  Generated: \"{first_token['token']}\" (expected: \"{expected}\")")
            print(f"  Iterations: {first_token['iterations']}")
            print(f"  Surprisal: {first_token['surprisal']:.2f} bits")
            print(f"  Frequency: {first_token['freq_category']}")


@torch.no_grad()
def evaluate_ppl_at_iterations(
    model: RecursiveTransformer,
    val_loader: DataLoader,
    max_iterations: int = 8,
    device: str = 'cuda',
    threshold: float = 0.5
) -> Dict[int, float]:
    """
    Evaluate perplexity at different max iteration caps.

    Unlike forcing a specific iteration count, this caps the maximum iterations
    while allowing the model to stop early via the done classifier.

    Returns dict mapping max iteration cap -> perplexity.
    """
    model.eval()
    results = {}
    original_max_iters = model.max_iterations

    for max_iters_cap in range(1, max_iterations + 1):
        total_loss = 0.0
        total_tokens = 0
        total_iters_used = 0.0

        # Temporarily cap max iterations
        model.max_iterations = max_iters_cap

        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Let done classifier decide when to stop (up to the cap)
            output, metadata = model(input_ids, threshold=threshold)

            # Compute loss (ignore padding)
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)),
                target_ids.view(-1),
                ignore_index=-100,
                reduction='sum'
            )

            # Count non-padding tokens
            valid_tokens = (target_ids != -100).sum().item()
            total_loss += loss.item()
            total_tokens += valid_tokens

            # Track average iterations actually used
            if 'iterations_per_position' in metadata:
                valid_mask = target_ids != -100
                total_iters_used += metadata['iterations_per_position'][valid_mask].sum().item()

        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        avg_iters = total_iters_used / total_tokens if total_tokens > 0 else max_iters_cap
        results[max_iters_cap] = ppl
        print(f"  max_iters={max_iters_cap}: PPL={ppl:.3f} (avg iters used: {avg_iters:.2f})")

    # Restore original max iterations
    model.max_iterations = original_max_iters

    return results


def plot_ppl_vs_iterations(
    results: Dict[int, float],
    save_path: Optional[str] = None,
    title: str = "Perplexity vs Number of Iterations"
):
    """Plot perplexity as a function of iteration count."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    iters = sorted(results.keys())
    ppls = [results[i] for i in iters]

    plt.figure(figsize=(10, 6))
    plt.plot(iters, ppls, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Number of Iterations', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(iters)

    # Add value labels
    for i, ppl in zip(iters, ppls):
        plt.annotate(f'{ppl:.2f}', (i, ppl), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # Highlight improvement
    if len(ppls) > 1:
        improvement = (ppls[0] - ppls[-1]) / ppls[0] * 100
        plt.text(0.95, 0.95, f'Improvement: {improvement:.1f}%\n({ppls[0]:.2f} → {ppls[-1]:.2f})',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def run_iteration_sweep(
    model: RecursiveTransformer,
    tokenizer,
    config: Dict,
    device: str = 'cuda',
    num_val_samples: int = 500,
    save_plot: Optional[str] = None,
    threshold: float = 0.5
):
    """
    Run full iteration sweep analysis.

    Evaluates model at each max iteration cap from 1 to max_iterations.
    The model can stop early via the done classifier, so this tests
    how performance changes as we allow more iterations.
    """
    print("\n" + "=" * 70)
    print("ITERATION SWEEP: PPL at Different Max Iteration Caps")
    print("=" * 70)

    # Create validation loader
    val_dataset = TinyStoriesDatasetFinite(
        tokenizer=tokenizer,
        max_seq_len=config.get('max_seq_len', 256),
        split="validation",
        max_samples=num_val_samples
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    max_iterations = config.get('max_iterations', 8)
    print(f"\nEvaluating PPL for max iterations 1 to {max_iterations} (threshold={threshold})...")

    results = evaluate_ppl_at_iterations(
        model, val_loader, max_iterations, device, threshold=threshold
    )

    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)

    best_iter = min(results, key=results.get)
    worst_iter = max(results, key=results.get)

    print(f"Best PPL:  {results[best_iter]:.3f} at {best_iter} iterations")
    print(f"Worst PPL: {results[worst_iter]:.3f} at {worst_iter} iterations")
    print(f"PPL at 1 iter:   {results[1]:.3f}")
    print(f"PPL at max iter: {results[max_iterations]:.3f}")

    improvement = (results[1] - results[max_iterations]) / results[1] * 100
    print(f"\nImprovement from 1 to {max_iterations} iterations: {improvement:.1f}%")

    # Check if more iterations helps
    if results[max_iterations] < results[1]:
        print("\n✓ More iterations HELPS - adaptive compute is beneficial!")
    else:
        print("\n✗ More iterations doesn't help - model may not be using iterations effectively")

    # Plot
    if save_plot:
        plot_ppl_vs_iterations(results, save_plot)
    else:
        plot_ppl_vs_iterations(results)

    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze iteration patterns in trained model')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--compute-frequencies', action='store_true',
                        help='Compute token frequencies from dataset (slow)')
    parser.add_argument('--num-prompts', type=int, default=10,
                        help='Number of prompts for comprehensive analysis')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt to analyze')
    parser.add_argument('--iteration-sweep', action='store_true',
                        help='Evaluate PPL at each iteration count (1 to max)')
    parser.add_argument('--sweep-samples', type=int, default=500,
                        help='Number of validation samples for iteration sweep')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save PPL vs iterations plot (e.g., ppl_sweep.png)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Done classifier threshold for iteration sweep (default: 0.5)')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, args.device)
    tokenizer = get_tokenizer()

    print(f"Model: d_model={config.get('d_model')}, n_layers={config.get('n_layers')}, "
          f"max_iterations={config.get('max_iterations')}")

    # Iteration sweep analysis
    if args.iteration_sweep:
        run_iteration_sweep(
            model, tokenizer, config,
            device=args.device,
            num_val_samples=args.sweep_samples,
            save_plot=args.save_plot,
            threshold=args.threshold
        )
        return

    # Compute frequencies if requested
    token_frequencies = None
    if args.compute_frequencies:
        token_frequencies = get_token_frequencies(tokenizer, num_samples=10000)
        print(f"Computed frequencies for {len(token_frequencies)} unique tokens")

    # Run analysis
    if args.prompt:
        # Single prompt analysis
        results = analyze_generation(
            model, tokenizer, args.prompt,
            max_tokens=50,
            token_frequencies=token_frequencies
        )
        print_generation_analysis(results, args.prompt)
        correlations = analyze_iteration_correlations(results)
        print(f"Correlation with surprisal: {correlations['correlation_with_surprisal']:.3f}")
    else:
        # Comprehensive analysis
        run_comprehensive_analysis(
            model, tokenizer,
            token_frequencies=token_frequencies,
            num_prompts=args.num_prompts
        )

        # Specific examples
        analyze_specific_examples(model, tokenizer, token_frequencies)


if __name__ == '__main__':
    main()
