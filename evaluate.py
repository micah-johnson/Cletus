"""
Evaluation and visualization utilities for Recursive Transformer.

Key questions to answer:
1. Does the model use more iterations for deeper expressions?
2. How do hidden states evolve across iterations?
3. What does the model attend to from previous iterations?
4. Does the done classifier learn to read "uncertainty"?
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from model import RecursiveTransformer
from dataset import ArithmeticTokenizer, NestedArithmeticDataset, generate_nested_arithmetic


class RecursiveTransformerAnalyzer:
    """Analysis and visualization tools for the recursive transformer."""

    def __init__(
        self,
        model: RecursiveTransformer,
        tokenizer: ArithmeticTokenizer,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def analyze_single_sample(
        self,
        expression: str,
        expected_result: Optional[int] = None,
        max_seq_len: int = 64,
        force_all_iterations: bool = False,
        threshold: float = 0.7
    ) -> Dict:
        """
        Run detailed analysis on a single expression.

        Args:
            force_all_iterations: If True, run all iterations (for analysis).
                                  If False, use threshold-based early stopping (like training).
            threshold: Done probability threshold for early stopping.

        Returns detailed info about iterations, hidden states, attention, etc.
        """
        # Prepare input - MUST pad to same length as training
        input_str = f"{expression} = "
        input_ids = self.tokenizer.encode(input_str)
        input_ids = self.tokenizer.pad_sequence(input_ids, max_seq_len)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Run model
        if force_all_iterations:
            output, metadata = self.model(
                input_tensor,
                threshold=0.0,
                return_all_states=True,
                force_iterations=self.model.max_iterations
            )
        else:
            output, metadata = self.model(
                input_tensor,
                threshold=threshold,
                return_all_states=True
            )

        # Get prediction - stop at first EOS token
        predicted_ids = output[0].argmax(dim=-1).tolist()

        # Find the first EOS token and truncate there
        try:
            eos_pos = predicted_ids.index(self.tokenizer.eos_token_id)
            predicted_ids = predicted_ids[:eos_pos + 1]
        except ValueError:
            pass  # No EOS found, use full sequence

        predicted_str = self.tokenizer.decode(predicted_ids)

        # Extract result from prediction (after '=')
        try:
            after_eq = predicted_str.split('=')[1].strip()
            # Extract just the number (possibly negative)
            match = re.match(r'^(-?\d+)', after_eq)
            pred_result = match.group(1) if match else "?"
        except:
            pred_result = "?"

        return {
            'expression': expression,
            'expected_result': expected_result,
            'predicted_result': pred_result,
            'input_str': input_str,
            'num_iterations': metadata['num_iterations'],
            'done_probs': metadata['done_probs'][0].cpu().numpy(),  # [n_iters, 1]
            'hidden_states': [s[0].cpu().numpy() for s in metadata['all_hidden_states']],
            'cross_attention': metadata['cross_attention_weights']
        }

    @torch.no_grad()
    def evaluate_iteration_scaling(
        self,
        test_data: List[Dict],
        threshold: float = 0.7,
        max_seq_len: int = 64
    ) -> Dict:
        """
        Key experiment: Does the model use more iterations for deeper expressions?

        Args:
            test_data: List of {'expression': str, 'result': int, 'depth': int}
            threshold: Done probability threshold
            max_seq_len: Sequence length to pad to (must match training)

        Returns:
            Results grouped by depth
        """
        results_by_depth = defaultdict(list)

        for sample in test_data:
            # Prepare input - pad to same length as training
            input_str = f"{sample['expression']} = "
            input_ids = self.tokenizer.encode(input_str)
            input_ids = self.tokenizer.pad_sequence(input_ids, max_seq_len)
            input_tensor = torch.tensor([input_ids], device=self.device)

            # Run model
            output, metadata = self.model(input_tensor, threshold=threshold)

            # Check if correct - stop at first EOS token
            predicted_ids = output[0].argmax(dim=-1).tolist()

            # Find the first EOS token and truncate there
            try:
                eos_pos = predicted_ids.index(self.tokenizer.eos_token_id)
                predicted_ids = predicted_ids[:eos_pos + 1]
            except ValueError:
                pass  # No EOS found, use full sequence

            predicted_str = self.tokenizer.decode(predicted_ids)

            try:
                after_eq = predicted_str.split('=')[1].strip()
                match = re.match(r'^(-?\d+)', after_eq)
                pred_result = int(match.group(1)) if match else None
                correct = (pred_result == sample['result'])
            except:
                correct = False

            results_by_depth[sample['depth']].append({
                'num_iterations': metadata['num_iterations'],
                'correct': correct,
                'done_probs': metadata['done_probs'][0].cpu().numpy(),
                'expression': sample['expression'],
                'expected': sample['result']
            })

        return dict(results_by_depth)

    def print_iteration_scaling_report(self, results_by_depth: Dict):
        """Print a formatted report of iteration scaling by depth."""
        print("\n" + "=" * 60)
        print("ITERATION SCALING ANALYSIS")
        print("=" * 60)
        print(f"{'Depth':<8} {'Samples':<10} {'Avg Iters':<12} {'Accuracy':<12}")
        print("-" * 60)

        depths = sorted(results_by_depth.keys())
        all_iters = []
        all_depths = []

        for depth in depths:
            items = results_by_depth[depth]
            avg_iter = np.mean([r['num_iterations'] for r in items])
            accuracy = np.mean([r['correct'] for r in items])

            print(f"{depth:<8} {len(items):<10} {avg_iter:<12.2f} {accuracy:<12.2%}")

            for r in items:
                all_iters.append(r['num_iterations'])
                all_depths.append(depth)

        # Compute correlation
        if len(all_depths) > 1:
            correlation = np.corrcoef(all_depths, all_iters)[0, 1]
            print("-" * 60)
            print(f"Depth-Iteration Correlation: {correlation:.3f}")

            if correlation > 0.3:
                print("-> Model uses MORE iterations for deeper expressions!")
            elif correlation < -0.3:
                print("-> Model uses FEWER iterations for deeper expressions (unexpected)")
            else:
                print("-> No strong relationship between depth and iterations")

        print("=" * 60 + "\n")

    def visualize_hidden_state_evolution(
        self,
        expression: str,
        save_path: Optional[str] = None,
        max_seq_len: int = 64
    ):
        """
        Visualize how hidden states change across iterations using PCA.

        This shows the "trajectory of thought" as the model iterates.
        """
        analysis = self.analyze_single_sample(expression, max_seq_len=max_seq_len, force_all_iterations=True)
        states = analysis['hidden_states']  # List of [seq_len, d_model]

        # Mean-pool each iteration's hidden state
        pooled = [s.mean(axis=0) for s in states]  # List of [d_model]
        pooled = np.stack(pooled)  # [n_iters, d_model]

        # PCA to 2D
        pca = PCA(n_components=2)
        projected = pca.fit_transform(pooled)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Trajectory plot
        ax = axes[0]
        ax.plot(projected[:, 0], projected[:, 1], 'b-', alpha=0.5, linewidth=2)
        scatter = ax.scatter(
            projected[:, 0], projected[:, 1],
            c=range(len(projected)),
            cmap='viridis',
            s=100,
            zorder=5
        )

        for i, (x, y) in enumerate(projected):
            ax.annotate(
                f'iter {i+1}',
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(f'Hidden State Trajectory\n{expression}')
        plt.colorbar(scatter, ax=ax, label='Iteration')

        # Right: Done probability over iterations
        ax = axes[1]
        done_probs = analysis['done_probs'].flatten()
        ax.bar(range(1, len(done_probs) + 1), done_probs, color='steelblue', alpha=0.7)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='threshold=0.5')
        ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='threshold=0.7')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Done Probability')
        ax.set_title('Done Classifier Confidence')
        ax.set_ylim(0, 1)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_iteration_comparison(
        self,
        results_by_depth: Dict,
        save_path: Optional[str] = None
    ):
        """
        Create visualization comparing iterations across depths.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        depths = sorted(results_by_depth.keys())

        # Left: Box plot of iterations by depth
        ax = axes[0]
        iteration_data = [
            [r['num_iterations'] for r in results_by_depth[d]]
            for d in depths
        ]
        bp = ax.boxplot(iteration_data, labels=depths, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
        ax.set_xlabel('Expression Depth')
        ax.set_ylabel('Number of Iterations')
        ax.set_title('Iterations by Expression Depth')

        # Right: Accuracy by depth
        ax = axes[1]
        accuracies = [
            np.mean([r['correct'] for r in results_by_depth[d]])
            for d in depths
        ]
        bars = ax.bar(depths, accuracies, color='forestgreen', alpha=0.7)
        ax.set_xlabel('Expression Depth')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Expression Depth')
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f'{acc:.1%}',
                ha='center',
                fontsize=10
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_done_probability_heatmap(
        self,
        test_data: List[Dict],
        save_path: Optional[str] = None,
        max_seq_len: int = 64
    ):
        """
        Heatmap showing done probability evolution for samples of each depth.
        """
        # Collect done probabilities grouped by depth
        done_probs_by_depth = defaultdict(list)

        for sample in test_data[:200]:  # Limit for visualization
            input_str = f"{sample['expression']} = "
            input_ids = self.tokenizer.encode(input_str)
            input_ids = self.tokenizer.pad_sequence(input_ids, max_seq_len)
            input_tensor = torch.tensor([input_ids], device=self.device)

            with torch.no_grad():
                _, metadata = self.model(
                    input_tensor,
                    threshold=0.0,
                    force_iterations=self.model.max_iterations
                )

            probs = metadata['done_probs'][0].cpu().numpy().flatten()
            done_probs_by_depth[sample['depth']].append(probs)

        # Create heatmap
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, depth in enumerate(sorted(done_probs_by_depth.keys())[:4]):
            ax = axes[idx]
            probs = np.array(done_probs_by_depth[depth])

            im = ax.imshow(probs, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Sample')
            ax.set_title(f'Depth {depth} - Done Probabilities')
            ax.set_xticks(range(probs.shape[1]))
            ax.set_xticklabels(range(1, probs.shape[1] + 1))

            plt.colorbar(im, ax=ax, label='P(done)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_training_history(
        self,
        history: Dict,
        save_path: Optional[str] = None
    ):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        ax = axes[0, 0]
        if 'train_loss' in history:
            ax.plot(history['train_loss'], label='Train', alpha=0.8)
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Task loss
        ax = axes[0, 1]
        if 'train_task_loss' in history:
            ax.plot(history['train_task_loss'], label='Train', alpha=0.8)
        if 'val_task_loss' in history:
            ax.plot(history['val_task_loss'], label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Task Loss')
        ax.set_title('Task Loss (Cross-Entropy)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Iterations used
        ax = axes[1, 0]
        if 'train_avg_iterations' in history:
            ax.plot(history['train_avg_iterations'], label='Train', alpha=0.8)
        if 'val_avg_iterations' in history:
            ax.plot(history['val_avg_iterations'], label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg Iterations')
        ax.set_title('Average Iterations Used')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Accuracy
        ax = axes[1, 1]
        if 'val_accuracy' in history:
            ax.plot(history['val_accuracy'], label='Val', color='green', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()


def run_full_evaluation(
    model: RecursiveTransformer,
    tokenizer: ArithmeticTokenizer,
    test_samples: int = 500,
    max_depth: int = 4,
    output_dir: str = 'results',
    device: str = 'cuda',
    max_seq_len: int = 64
):
    """
    Run complete evaluation suite on a trained model.

    Args:
        max_seq_len: Sequence length to pad to (must match training config)
    """
    os.makedirs(output_dir, exist_ok=True)

    analyzer = RecursiveTransformerAnalyzer(model, tokenizer, device)

    # Generate test data
    print("Generating test data...")
    test_data = []
    samples_per_depth = test_samples // max_depth
    for depth in range(1, max_depth + 1):
        for _ in range(samples_per_depth):
            sample = generate_nested_arithmetic(max_depth=max_depth, target_depth=depth)
            test_data.append(sample)

    # 1. Iteration scaling analysis
    print("\nRunning iteration scaling analysis...")
    results = analyzer.evaluate_iteration_scaling(test_data, threshold=0.7, max_seq_len=max_seq_len)
    analyzer.print_iteration_scaling_report(results)

    # 2. Visualizations
    print("Generating visualizations...")

    # Iteration comparison
    analyzer.visualize_iteration_comparison(
        results,
        save_path=os.path.join(output_dir, 'iteration_comparison.png')
    )

    # Done probability heatmap
    analyzer.visualize_done_probability_heatmap(
        test_data,
        save_path=os.path.join(output_dir, 'done_probability_heatmap.png'),
        max_seq_len=max_seq_len
    )

    # Sample trajectories for each depth
    for depth in range(1, min(max_depth + 1, 5)):
        sample = generate_nested_arithmetic(max_depth=max_depth, target_depth=depth)
        analyzer.visualize_hidden_state_evolution(
            sample['expression'],
            save_path=os.path.join(output_dir, f'trajectory_depth{depth}.png'),
            max_seq_len=max_seq_len
        )

    print(f"\nResults saved to {output_dir}/")
    return results


if __name__ == '__main__':
    # Quick test of visualization
    print("Testing evaluation utilities...")

    # Create sample data
    tokenizer = ArithmeticTokenizer()

    # Generate test samples
    test_data = []
    for depth in range(1, 5):
        for _ in range(50):
            test_data.append(generate_nested_arithmetic(max_depth=4, target_depth=depth))

    print(f"Generated {len(test_data)} test samples")

    # Would need a trained model to run full evaluation
    print("\nTo run full evaluation, load a trained model and call run_full_evaluation()")
