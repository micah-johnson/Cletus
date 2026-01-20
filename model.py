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
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Any


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final number from text for answer comparison.

    Examples:
        "The answer is 8" → "8"
        "The answer is -42" → "-42"
        "8" → "8"
        "No number here" → None
    """
    nums = re.findall(r'-?\d+', text)
    return nums[-1] if nums else None


def check_answer_correct(pred_text: str, target_text: str) -> bool:
    """Check if predicted and target answers match (number-only comparison)."""
    pred_num = extract_answer(pred_text)
    target_num = extract_answer(target_text)
    if pred_num is None or target_num is None:
        return False
    return pred_num == target_num


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
            # Shape: [batch, num_prev_iters * seq_len, d_model]
            prev_concat = torch.cat(previous_states, dim=1)

            # CRITICAL: Apply causal mask to cross-attention too!
            # Position i can only attend to positions 0..i from each previous iteration
            # For multiple iterations, we tile the mask
            num_prev_iters = len(previous_states)
            seq_len = x.size(1)
            if attn_mask is not None:
                # Tile the causal mask for each previous iteration
                # attn_mask is [seq_len, seq_len], we need [seq_len, num_prev_iters * seq_len]
                cross_attn_mask = attn_mask.repeat(1, num_prev_iters)
            else:
                cross_attn_mask = None

            cross_out, cross_attn_weights = self.cross_attn(
                x_norm, prev_concat, prev_concat,
                attn_mask=cross_attn_mask,
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
    Predicts whether each position is "done thinking" based on hidden states.
    Per-position classification allows different tokens to use different iteration counts.

    Easy tokens ("The", "is") → done after 1 iteration
    Hard tokens (computed answers) → need more iterations
    """

    def __init__(self, d_model: int):
        super().__init__()
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
            done_logits: [batch_size, seq_len] logit per position
        """
        # Apply MLP to each position independently
        done_logits = self.mlp(hidden_states).squeeze(-1)  # [batch, seq_len]
        return done_logits


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

        # Tie input and output embeddings (reduces params by ~50%)
        self.output_head.weight = self.embedding.weight

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
        Forward pass with per-position done classification.

        Args:
            input_ids: [batch_size, seq_len] input token IDs
            attention_mask: [batch_size, seq_len] attention mask (1 = attend, 0 = ignore)
            threshold: Done probability threshold for early stopping (inference only)
            return_all_states: Whether to return hidden states from all iterations
            force_iterations: If set, run exactly this many iterations
            detach_hidden: If True, detach hidden states (saves memory)

        Returns:
            output: [batch_size, seq_len, vocab_size] logits
            metadata: Dict with per-position iteration info
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token + position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)

        # Create causal mask - prevent attending to future tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        all_hidden_states = []
        all_done_logits = []  # [num_iters] of [batch, seq_len]
        all_done_probs = []
        all_outputs = []  # [num_iters] of [batch, seq_len, vocab]
        cross_attention_weights = []

        for iteration in range(self.max_iterations):
            # Add iteration embedding (broadcast to all positions)
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

            # Compute output at this iteration (for all positions)
            iter_output = self.output_head(self.output_norm(x_iter))
            all_outputs.append(iter_output)

            # Per-position done classification
            done_logits = self.done_classifier(x_iter)  # [batch, seq_len]
            all_done_logits.append(done_logits)
            done_probs = torch.sigmoid(done_logits)  # [batch, seq_len]
            all_done_probs.append(done_probs)

            # Update x for next iteration
            x = x_iter

            # Early stopping conditions
            if force_iterations is not None:
                if iteration + 1 >= force_iterations:
                    break
            elif not self.training:
                # During inference: stop when all positions are done
                if done_probs.min() > threshold:
                    break

        num_iterations = iteration + 1

        # Stack per-iteration tensors: [batch, num_iters, seq_len]
        stacked_done_logits = torch.stack(all_done_logits, dim=1)
        stacked_done_probs = torch.stack(all_done_probs, dim=1)

        # Compute per-position iteration counts (when each position first crossed threshold)
        # Shape: [batch, seq_len]
        with torch.no_grad():
            done_mask = stacked_done_probs > threshold  # [batch, num_iters, seq_len]
            # Find first iteration where done for each position
            # If never done, use num_iterations
            first_done = torch.argmax(done_mask.int(), dim=1) + 1  # +1 for 1-indexed
            never_done = ~done_mask.any(dim=1)
            first_done[never_done] = num_iterations
            iterations_per_position = first_done.float()

        # Final output from last iteration
        output = all_outputs[-1]

        metadata = {
            'num_iterations': num_iterations,
            'done_logits': stacked_done_logits,  # [batch, num_iters, seq_len]
            'done_probs': stacked_done_probs,    # [batch, num_iters, seq_len]
            'iterations_per_position': iterations_per_position,  # [batch, seq_len]
            'all_outputs': all_outputs,
            'all_hidden_states': all_hidden_states if return_all_states else None,
            'cross_attention_weights': cross_attention_weights if return_all_states else None
        }

        return output, metadata

    def get_gate_values(self) -> List[float]:
        """Return current cross-attention gate values for each layer."""
        return [torch.sigmoid(layer.cross_attn_gate).item() for layer in self.layers]

    @torch.no_grad()
    def generate_token(
        self,
        input_ids: torch.Tensor,
        threshold: float = 0.5,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        cached_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, int, Optional[List[torch.Tensor]]]:
        """
        Generate a single token with per-position early exit.

        Only checks done classifier on the LAST position, allowing efficient
        generation where earlier positions can reuse computation.

        Args:
            input_ids: [batch_size, seq_len] current sequence
            threshold: Done probability threshold for early stopping
            temperature: Sampling temperature
            top_k: If set, sample from top-k tokens only
            cached_states: Optional cached hidden states from previous calls

        Returns:
            next_token: [batch_size, 1] sampled token
            iterations_used: Number of iterations run
            new_cached_states: Updated cached states for next call
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token + position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        all_hidden_states = cached_states if cached_states else []
        iterations_used = 0

        for iteration in range(self.max_iterations):
            # Add iteration embedding
            iter_emb = self.iteration_embedding(
                torch.full((batch_size,), iteration, device=device, dtype=torch.long)
            ).unsqueeze(1)
            x_iter = x + iter_emb

            # Forward through layers
            for layer in self.layers:
                x_iter, _ = layer(
                    x_iter,
                    previous_states=all_hidden_states if all_hidden_states else None,
                    attn_mask=causal_mask,
                    return_attention=False
                )

            all_hidden_states.append(x_iter)
            x = x_iter
            iterations_used = iteration + 1

            # Check done for LAST position only
            done_logits = self.done_classifier(x_iter)  # [batch, seq_len]
            done_prob_last = torch.sigmoid(done_logits[:, -1])  # [batch]

            if done_prob_last.min() > threshold:
                break

        # Get logits for last position
        last_hidden = x[:, -1:, :]  # [batch, 1, d_model]
        logits = self.output_head(self.output_norm(last_hidden))  # [batch, 1, vocab]
        logits = logits.squeeze(1)  # [batch, vocab]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]

        return next_token, iterations_used, all_hidden_states

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        threshold: float = 0.5,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Autoregressive generation with per-token iteration tracking.

        Args:
            input_ids: [batch_size, prompt_len] prompt tokens
            max_new_tokens: Maximum tokens to generate
            threshold: Done probability threshold
            temperature: Sampling temperature
            top_k: Top-k sampling
            eos_token_id: Stop when this token is generated
            pad_token_id: Token to use for padding

        Returns:
            output_ids: [batch_size, prompt_len + generated_len] full sequence
            iterations_per_token: List of iteration counts for each generated token
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        current_ids = input_ids.clone()
        iterations_per_token = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Generate one token (no caching for simplicity - can add later)
            next_token, iters_used, _ = self.generate_token(
                current_ids,
                threshold=threshold,
                temperature=temperature,
                top_k=top_k,
                cached_states=None
            )

            iterations_per_token.append(iters_used)

            # Handle finished sequences
            if eos_token_id is not None:
                just_finished = (next_token.squeeze(-1) == eos_token_id)
                finished = finished | just_finished

                if pad_token_id is not None:
                    next_token[finished.unsqueeze(-1).expand_as(next_token)] = pad_token_id

            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Check if all sequences finished
            if finished.all():
                break

        return current_ids, iterations_per_token


def compute_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    metadata: Dict,
    iteration_cost: float = 0.01,
    done_supervision_weight: float = 0.5,
    pad_token_id: Optional[int] = None,
    tokenizer: Any = None
) -> Tuple[torch.Tensor, Dict]:
    """
    Per-position loss with done classifier supervision.

    Loss components:
    1. Task loss: Cross-entropy on final output
    2. Done supervision: Per-position BCE loss - position learns when ITS token is correct
    3. Iteration cost: Penalty for positions using more iterations

    The done classifier learns per-token:
    - "The" → fires done at iter 1 (easy token)
    - "8" (computed) → fires done at iter 4 (needed more thinking)

    Args:
        tokenizer: Optional - if provided, computes number-only accuracy for evaluation
    """
    device = output.device
    batch_size, seq_len = output.size(0), output.size(1)
    all_outputs = metadata['all_outputs']
    done_logits = metadata['done_logits']  # [batch, num_iters, seq_len]
    done_probs = metadata['done_probs']    # [batch, num_iters, seq_len]
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

    # 2. Per-position done classifier supervision
    # For each position at each iteration: is the prediction correct?
    # Shape: [batch, num_iters, seq_len]
    per_position_correct = []
    per_iter_accuracies = []

    for iter_idx, iter_output in enumerate(all_outputs):
        predictions = iter_output.argmax(dim=-1)  # [batch, seq_len]
        correct = (predictions == target).float()  # [batch, seq_len]

        # For overall accuracy, we care about answer positions
        if pad_token_id is not None:
            answer_mask = (target != pad_token_id)
            seq_correct = ((predictions == target) | ~answer_mask).all(dim=-1).float()
        else:
            seq_correct = (predictions == target).all(dim=-1).float()

        per_position_correct.append(correct)
        per_iter_accuracies.append(seq_correct.mean().item())

    # Stack: [batch, num_iters, seq_len]
    done_targets = torch.stack(per_position_correct, dim=1)

    # Cumulative max over iterations: once correct, stays "done"
    done_targets_cummax, _ = torch.cummax(done_targets, dim=1)

    # Create mask for valid positions (non-padding in target)
    if pad_token_id is not None:
        valid_mask = (target != pad_token_id).unsqueeze(1).expand(-1, num_iters, -1)
    else:
        valid_mask = torch.ones_like(done_targets_cummax, dtype=torch.bool)

    # BCE loss only on valid positions
    done_supervision_loss = F.binary_cross_entropy_with_logits(
        done_logits,
        done_targets_cummax,
        reduction='none'
    )
    done_supervision_loss = (done_supervision_loss * valid_mask.float()).sum() / valid_mask.float().sum()

    # 3. Per-position iteration cost
    # Penalize positions for "continuing" (not being done)
    # Expected iterations per position = sum over iterations of (1 - done_prob)
    continuation_probs = 1 - done_probs  # [batch, num_iters, seq_len]
    if pad_token_id is not None:
        # Only count answer positions
        answer_mask = (target != pad_token_id).unsqueeze(1)  # [batch, 1, seq_len]
        expected_iters_per_pos = (continuation_probs * answer_mask.float()).sum(dim=1)  # [batch, seq_len]
        iter_loss = iteration_cost * expected_iters_per_pos[answer_mask.squeeze(1)].mean()
    else:
        expected_iters_per_pos = continuation_probs.sum(dim=1)  # [batch, seq_len]
        iter_loss = iteration_cost * expected_iters_per_pos.mean()

    # Total loss
    total_loss = task_loss + done_supervision_weight * done_supervision_loss + iter_loss

    # Compute average iterations per position (for logging)
    with torch.no_grad():
        # Use the precomputed iterations_per_position if available
        if 'iterations_per_position' in metadata:
            iters_per_pos = metadata['iterations_per_position']
            if pad_token_id is not None:
                answer_mask = (target != pad_token_id)
                avg_iters = iters_per_pos[answer_mask].mean().item() if answer_mask.any() else num_iters
            else:
                avg_iters = iters_per_pos.mean().item()
        else:
            avg_iters = num_iters

    # Compute number-only accuracy if tokenizer provided (for evaluation)
    number_accuracy = None
    if tokenizer is not None:
        with torch.no_grad():
            predictions = output.argmax(dim=-1)  # [batch, seq_len]
            num_correct = 0
            for i in range(batch_size):
                # Decode prediction and target (answer portion only)
                if pad_token_id is not None:
                    answer_mask = (target[i] != pad_token_id)
                    pred_tokens = predictions[i][answer_mask].tolist()
                    tgt_tokens = target[i][answer_mask].tolist()
                else:
                    pred_tokens = predictions[i].tolist()
                    tgt_tokens = target[i].tolist()

                pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                tgt_text = tokenizer.decode(tgt_tokens, skip_special_tokens=True)

                if check_answer_correct(pred_text, tgt_text):
                    num_correct += 1

            number_accuracy = num_correct / batch_size

    metrics = {
        'task_loss': task_loss.item(),
        'done_loss': done_supervision_loss.item(),
        'iter_loss': iter_loss.item(),
        'avg_iterations': avg_iters,
        'max_iterations_run': num_iters,
        'mean_done_prob': done_probs[:, -1, :].mean().item(),
        'per_iter_accuracy': per_iter_accuracies,
        'final_accuracy': per_iter_accuracies[-1] if per_iter_accuracies else 0.0
    }

    if number_accuracy is not None:
        metrics['number_accuracy'] = number_accuracy

    return total_loss, metrics


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

        # Tie input and output embeddings (same as RecursiveTransformer)
        self.output_head.weight = self.embedding.weight

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

        # Causal mask - prevent attending to future tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        for repeat in range(self.n_repeats):
            for layer in self.layers:
                x = layer(x, attn_mask=causal_mask)

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

        # Tie input and output embeddings
        self.output_head.weight = self.embedding.weight

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

        # Causal mask - prevent attending to future tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask)

        output = self.output_head(self.output_norm(x))

        metadata = {
            'num_iterations': 1,
            'n_layers': self.n_layers,
        }

        return output, metadata

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
