"""
Recursive Transformer for Language Tasks

A transformer that can "think again" by letting each forward pass attend to
hidden states from previous passes. Designed for language tasks like GSM8K.

Uses standard PyTorch modules:
- nn.MultiheadAttention for attention
- nn.Linear + GELU for FFN
- nn.Embedding for token embeddings
- RMSNorm for normalization (Llama-style)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used by Llama)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - used by Llama."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos and sin for all positions
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int):
        """Precompute cos and sin values for efficiency."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer('cos_cached', emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        if seq_len > self.max_seq_len:
            self._precompute_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RecursiveTransformerLayer(nn.Module):
    """
    Transformer layer with cross-attention to previous iterations.

    Architecture:
        1. Self-attention with RoPE (standard)
        2. Cross-attention to previous iterations' hidden states (gated)
        3. Feedforward (Linear + GELU + Linear)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Self-attention using standard MultiheadAttention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = RMSNorm(d_model)

        # Cross-attention to previous iterations
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = RMSNorm(d_model)

        # Feedforward: Linear + GELU + Linear
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = RMSNorm(d_model)

        # Learnable gate for cross-attention (initialized to 0 - starts ignoring)
        self.cross_attn_gate = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        previous_states: Optional[List[torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            previous_states: List of tensors from previous iterations
            attn_mask: Causal mask for self-attention
            return_attention: Whether to return cross-attention weights

        Returns:
            x: Updated hidden states
            cross_attn_weights: Optional attention weights
        """
        cross_attn_weights = None

        # Self-attention with pre-norm
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask
        )
        x = residual + self.dropout(attn_out)

        # Cross-attention to previous iterations (if any)
        if previous_states is not None and len(previous_states) > 0:
            residual = x
            x_norm = self.norm2(x)

            # Concatenate all previous hidden states
            prev_concat = torch.cat(previous_states, dim=1)

            cross_out, cross_attn_weights = self.cross_attn(
                x_norm, prev_concat, prev_concat,
                need_weights=return_attention
            )

            # Gated addition - model learns to use or ignore cross-attention
            gate = torch.sigmoid(self.cross_attn_gate)
            x = residual + gate * self.dropout(cross_out)

        # Feedforward with pre-norm
        residual = x
        x_norm = self.norm3(x)
        ff_out = self.ff(x_norm)
        x = residual + ff_out

        return x, cross_attn_weights


class DoneClassifier(nn.Module):
    """
    Predicts whether the model is "done thinking" based on current hidden states.
    Learns to read "uncertainty" or "incompleteness" in the hidden representations.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
            # No Sigmoid - return logits for numerical stability
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, d_model]
        Returns:
            done_logit: [batch_size, 1] logit for "done thinking"
        """
        pooled = self.pool(hidden_states.transpose(1, 2)).squeeze(-1)
        done_logit = self.mlp(pooled)
        return done_logit


class RecursiveTransformer(nn.Module):
    """
    Recursive Transformer for language tasks.

    Key features:
    - Each forward pass can attend to all previous passes' hidden states
    - A learned "done" classifier decides when to stop iterating
    - Iteration embeddings tell the model which pass it's on
    - Uses RMSNorm and standard PyTorch modules
    - Designed for Llama tokenizer (32k vocab)
    """

    def __init__(
        self,
        vocab_size: int = 32000,  # Llama vocab size
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_iterations: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()

        self.d_model = d_model
        self.max_iterations = max_iterations
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Learned position embeddings (simpler than RoPE for this use case)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers with cross-attention capability
        self.layers = nn.ModuleList([
            RecursiveTransformerLayer(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])

        # Done classifier - learns when to stop
        self.done_classifier = DoneClassifier(d_model)

        # Output projection
        self.output_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        # Iteration embedding - tells the model which iteration it's on
        self.iteration_embedding = nn.Embedding(max_iterations, d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        return_all_states: bool = False,
        force_iterations: Optional[int] = None,
        detach_hidden: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            input_ids: [batch_size, seq_len] input token IDs
            attention_mask: [batch_size, seq_len] attention mask (1 = attend, 0 = ignore)
            threshold: Done probability threshold for early stopping
            return_all_states: Whether to return hidden states from all iterations
            force_iterations: If set, run exactly this many iterations
            detach_hidden: If True, detach hidden states (saves memory)

        Returns:
            output: [batch_size, seq_len, vocab_size] logits
            metadata: Dict with iteration info, done probabilities, etc.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token + position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)

        # Create causal mask if not provided
        # For autoregressive generation, we need causal masking
        causal_mask = None  # Let PyTorch handle causal attention with is_causal=True

        all_hidden_states = []
        done_logits = []
        done_probs = []
        all_outputs = []
        cross_attention_weights = []

        for iteration in range(self.max_iterations):
            # Add iteration embedding
            iter_emb = self.iteration_embedding(
                torch.full((batch_size,), iteration, device=device, dtype=torch.long)
            ).unsqueeze(1)
            x_iter = x + iter_emb

            # Forward through layers with cross-attention to previous iterations
            layer_cross_attns = []
            for layer in self.layers:
                x_iter, cross_attn = layer(
                    x_iter,
                    previous_states=all_hidden_states if all_hidden_states else None,
                    attn_mask=causal_mask,
                    return_attention=return_all_states
                )
                if cross_attn is not None:
                    layer_cross_attns.append(cross_attn)

            if layer_cross_attns:
                cross_attention_weights.append(layer_cross_attns)

            # Store this iteration's hidden states
            if detach_hidden:
                all_hidden_states.append(x_iter.detach().clone())
            else:
                all_hidden_states.append(x_iter)

            # Compute output at this iteration
            iter_output = self.output_head(self.output_norm(x_iter))
            all_outputs.append(iter_output)

            # Check if done
            done_logit = self.done_classifier(x_iter)
            done_logits.append(done_logit)
            done_prob = torch.sigmoid(done_logit)
            done_probs.append(done_prob)

            # Update x for potential next iteration
            x = x_iter

            # Early stopping conditions
            if force_iterations is not None:
                if iteration + 1 >= force_iterations:
                    break
            elif not self.training and done_prob.mean() > threshold:
                break

        # Final output from last iteration
        output = all_outputs[-1]

        metadata = {
            'num_iterations': iteration + 1,
            'done_logits': torch.stack(done_logits, dim=1),
            'done_probs': torch.stack(done_probs, dim=1),
            'all_outputs': all_outputs,
            'all_hidden_states': all_hidden_states if return_all_states else None,
            'cross_attention_weights': cross_attention_weights if return_all_states else None
        }

        return output, metadata

    def get_gate_values(self) -> List[float]:
        """Return current cross-attention gate values for each layer."""
        return [torch.sigmoid(layer.cross_attn_gate).item() for layer in self.layers]


def compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    metadata: Dict,
    iteration_cost: float = 0.01,
    done_supervision_weight: float = 0.5,
    pad_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Combined loss function with done classifier supervision.

    Loss components:
    1. Task loss: Cross-entropy on final output
    2. Done supervision: BCE loss teaching done classifier when answer is ready
    3. Iteration cost: Small penalty for using more iterations
    """
    device = output.device
    batch_size = output.size(0)
    all_outputs = metadata['all_outputs']
    done_logits = metadata['done_logits']
    done_probs = metadata['done_probs']
    num_iters = len(all_outputs)

    # 1. Task loss on final output
    if pad_token_id is not None:
        task_loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
            ignore_index=pad_token_id,
            reduction='mean'
        )
    else:
        task_loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
            reduction='mean'
        )

    # 2. Done classifier supervision
    done_targets = []
    per_iter_accuracies = []

    for iter_idx, iter_output in enumerate(all_outputs):
        predictions = iter_output.argmax(dim=-1)

        if pad_token_id is not None:
            mask = (target != pad_token_id)
            correct = ((predictions == target) | ~mask).all(dim=-1).float()
        else:
            correct = (predictions == target).all(dim=-1).float()

        done_targets.append(correct)
        per_iter_accuracies.append(correct.mean().item())

    done_targets = torch.stack(done_targets, dim=1)
    done_targets_cummax, _ = torch.cummax(done_targets, dim=1)

    done_logits_squeezed = done_logits.squeeze(-1)[:, :num_iters]
    done_probs_squeezed = done_probs.squeeze(-1)[:, :num_iters]
    done_targets_cummax = done_targets_cummax[:, :num_iters]

    done_supervision_loss = F.binary_cross_entropy_with_logits(
        done_logits_squeezed,
        done_targets_cummax,
        reduction='mean'
    )

    # 3. Iteration cost
    continuation_probs = 1 - done_probs_squeezed
    expected_extra_iters = continuation_probs.sum(dim=1).mean()
    iter_loss = iteration_cost * expected_extra_iters

    # Total loss
    total_loss = task_loss + done_supervision_weight * done_supervision_loss + iter_loss

    return total_loss, {
        'task_loss': task_loss.item(),
        'done_loss': done_supervision_loss.item(),
        'iter_loss': iter_loss.item(),
        'avg_iterations': num_iters,
        'mean_done_prob': done_probs[:, -1].mean().item(),
        'per_iter_accuracy': per_iter_accuracies,
        'final_accuracy': per_iter_accuracies[-1] if per_iter_accuracies else 0.0
    }


# =============================================================================
# BASELINE: Standard Weight-Tied Transformer (no cross-attention, no done classifier)
# =============================================================================

class StandardTransformerLayer(nn.Module):
    """Standard transformer layer WITHOUT cross-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = RMSNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = residual + ff_out

        return x


class StandardWeightTiedTransformer(nn.Module):
    """Baseline: Standard weight-tied transformer for comparison."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        n_repeats: int = 3,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_repeats = n_repeats
        self.effective_depth = n_layers * n_repeats
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            StandardTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)

        for repeat in range(self.n_repeats):
            for layer in self.layers:
                x = layer(x)

        output = self.output_head(self.output_norm(x))

        metadata = {
            'num_iterations': self.n_repeats,
            'effective_depth': self.effective_depth,
        }

        return output, metadata

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_loss_standard(
    output: torch.Tensor,
    target: torch.Tensor,
    pad_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, Dict]:
    """Simple cross-entropy loss for standard transformer."""
    if pad_token_id is not None:
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
            ignore_index=pad_token_id,
            reduction='mean'
        )
    else:
        loss = F.cross_entropy(
            output.view(-1, output.size(-1)),
            target.view(-1),
            reduction='mean'
        )

    predictions = output.argmax(dim=-1)
    if pad_token_id is not None:
        mask = (target != pad_token_id)
        correct = ((predictions == target) | ~mask).all(dim=-1).float()
    else:
        correct = (predictions == target).all(dim=-1).float()

    return loss, {
        'task_loss': loss.item(),
        'accuracy': correct.mean().item()
    }


class StandardTransformer(nn.Module):
    """Standard transformer - no weight tying, no recursion."""

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            StandardTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        output = self.output_head(self.output_norm(x))

        metadata = {
            'num_iterations': 1,
            'n_layers': self.n_layers,
        }

        return output, metadata

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
