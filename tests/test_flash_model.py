"""
Unit tests for FlashRecursiveTransformer and FlashRecursiveTransformerLayer.

Run with: pytest tests/test_flash_model.py -v
"""

import torch
import pytest
from model import FlashRecursiveTransformer, FlashRecursiveTransformerLayer, DoneClassifier


class TestFlashRecursiveTransformerLayer:
    """Tests for the FlashRecursiveTransformerLayer."""

    @pytest.fixture
    def layer(self):
        return FlashRecursiveTransformerLayer(
            d_model=128,
            n_heads=4,
            d_ff=512,
            dropout=0.0,
            max_seq_len=64
        )

    def test_layer_shapes(self, layer):
        """Verify output shapes match input shapes."""
        batch_size, seq_len, d_model = 2, 32, 128

        x = torch.randn(batch_size, seq_len, d_model)
        out = layer(x, prev_state=None, is_causal=True)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_layer_with_prev_state(self, layer):
        """Test layer with previous state (cross-attention)."""
        batch_size, seq_len, d_model = 2, 32, 128

        x = torch.randn(batch_size, seq_len, d_model)
        prev_state = torch.randn(batch_size, seq_len, d_model)

        out = layer(x, prev_state=prev_state, is_causal=True)

        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_cross_attention_gate_initialization(self, layer):
        """Cross-attention gate should be initialized to 0 (starts ignoring)."""
        assert layer.cross_attn_gate.item() == 0.0

    def test_gradients_flow(self, layer):
        """Verify gradients flow through layer."""
        batch_size, seq_len, d_model = 2, 16, 128

        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        prev_state = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        out = layer(x, prev_state=prev_state, is_causal=True)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert prev_state.grad is not None, "Gradients should flow to prev_state"
        assert x.grad.abs().sum() > 0, "Input gradients should be non-zero"
        assert prev_state.grad.abs().sum() > 0, "prev_state gradients should be non-zero"


class TestFlashRecursiveTransformer:
    """Tests for the FlashRecursiveTransformer model."""

    @pytest.fixture
    def model(self):
        return FlashRecursiveTransformer(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_iterations=4,
            dropout=0.0,
            max_seq_len=64
        )

    def test_forward_pass(self, model):
        """Run full forward pass and verify output structure."""
        batch_size, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch_size, seq_len))

        output, metadata = model(x, force_iterations=4)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 1000), f"Output shape mismatch"

        # Check metadata structure
        assert 'num_iterations' in metadata
        assert 'done_logits' in metadata
        assert 'done_probs' in metadata
        assert 'iterations_per_position' in metadata
        assert 'all_outputs' in metadata

        # Check metadata shapes
        assert metadata['num_iterations'] == 4
        assert metadata['done_logits'].shape == (batch_size, 4, seq_len)
        assert metadata['done_probs'].shape == (batch_size, 4, seq_len)
        assert metadata['iterations_per_position'].shape == (batch_size, seq_len)
        assert len(metadata['all_outputs']) == 4

    def test_backward_pass(self, model):
        """Verify gradients flow through all iterations."""
        batch_size, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch_size, seq_len))

        output, metadata = model(x, force_iterations=4)
        loss = output.sum()
        loss.backward()

        # Verify embedding gradients exist
        assert model.embedding.weight.grad is not None
        assert model.embedding.weight.grad.abs().sum() > 0

        # Verify iteration embedding gradients
        assert model.iteration_embedding.weight.grad is not None
        assert model.iteration_embedding.weight.grad.abs().sum() > 0

    def test_gradient_chain(self, model):
        """Verify grad(loss, iteration_0_embedding) is non-zero.

        This confirms gradients flow through all iterations back to the start.
        """
        batch_size, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch_size, seq_len))

        model.zero_grad()
        output, metadata = model(x, force_iterations=4)
        loss = output.sum()
        loss.backward()

        # Check that iteration 0 embedding has gradients
        # This means gradients flowed all the way back through 4 iterations
        iter_0_grad = model.iteration_embedding.weight.grad[0]
        assert iter_0_grad.abs().sum() > 0, "Iteration 0 embedding should have non-zero gradients"

    def test_early_stopping_inference(self, model):
        """Test early stopping during inference."""
        batch_size, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            output, metadata = model(x, threshold=0.5)

        # Should run at least 1 iteration
        assert metadata['num_iterations'] >= 1

    def test_force_iterations(self, model):
        """Test force_iterations parameter."""
        batch_size, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch_size, seq_len))

        for force_iters in [1, 2, 3, 4]:
            output, metadata = model(x, force_iterations=force_iters)
            assert metadata['num_iterations'] == force_iters

    def test_detach_hidden(self, model):
        """Test detach_hidden parameter."""
        batch_size, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch_size, seq_len))

        output, metadata = model(x, force_iterations=2, detach_hidden=True, return_all_states=True)

        # Hidden states should be detached (no grad_fn)
        for state in metadata['all_hidden_states']:
            assert not state.requires_grad

    def test_return_all_states(self, model):
        """Test return_all_states parameter."""
        batch_size, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch_size, seq_len))

        # Without return_all_states
        output, metadata = model(x, force_iterations=2, return_all_states=False)
        assert metadata['all_hidden_states'] is None

        # With return_all_states
        output, metadata = model(x, force_iterations=2, return_all_states=True)
        assert metadata['all_hidden_states'] is not None
        assert len(metadata['all_hidden_states']) == 2

    def test_get_gate_values(self, model):
        """Test get_gate_values method."""
        gates = model.get_gate_values()

        assert len(gates) == 2  # 2 layers
        for gate in gates:
            assert 0 <= gate <= 1  # Should be sigmoid output


class TestFlashModelVsOriginal:
    """Tests comparing Flash model to original RecursiveTransformer."""

    def test_same_interface(self):
        """Flash model should have the same interface as original."""
        from model import RecursiveTransformer

        # Create both models
        flash_model = FlashRecursiveTransformer(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_iterations=4,
            dropout=0.0,
            max_seq_len=64
        )

        original_model = RecursiveTransformer(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_iterations=4,
            dropout=0.0,
            max_seq_len=64
        )

        x = torch.randint(0, 1000, (2, 32))

        # Both should work with same inputs
        flash_out, flash_meta = flash_model(x, force_iterations=2)
        original_out, original_meta = original_model(x, force_iterations=2)

        # Same output shape
        assert flash_out.shape == original_out.shape

        # Same metadata keys
        assert set(flash_meta.keys()) == set(original_meta.keys())

    def test_generate_interface(self):
        """Flash model generate should work the same as original."""
        flash_model = FlashRecursiveTransformer(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            max_iterations=4,
            dropout=0.0,
            max_seq_len=64
        )

        x = torch.randint(0, 1000, (1, 10))

        flash_model.eval()
        output_ids, iterations_per_token = flash_model.generate(
            x,
            max_new_tokens=5,
            temperature=1.0,
            top_k=50
        )

        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] >= 10  # At least prompt length
        assert len(iterations_per_token) > 0


class TestCausalMasking:
    """Tests for causal masking in attention."""

    def test_causal_self_attention(self):
        """Verify self-attention is causal (no future attention)."""
        layer = FlashRecursiveTransformerLayer(
            d_model=128,
            n_heads=4,
            d_ff=512,
            dropout=0.0,
            max_seq_len=64
        )

        batch_size, seq_len, d_model = 1, 16, 128

        # Create input where position 0 has distinct values
        x = torch.randn(batch_size, seq_len, d_model)
        x[:, 0, :] = 100.0  # Make position 0 very different

        out1 = layer(x, prev_state=None, is_causal=True)

        # Now change position 1 onwards
        x2 = x.clone()
        x2[:, 1:, :] = torch.randn(batch_size, seq_len - 1, d_model)

        out2 = layer(x2, prev_state=None, is_causal=True)

        # Position 0 output should be the same (can't attend to future)
        assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5), \
            "Position 0 output should not depend on future positions"


class TestMemoryEfficiency:
    """Tests for memory efficiency of Flash model."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_vs_iterations(self):
        """Memory should scale linearly with iterations (O(N) not O(N²))."""
        model = FlashRecursiveTransformer(
            vocab_size=1000,
            d_model=256,
            n_heads=8,
            n_layers=4,
            d_ff=1024,
            max_iterations=8,
            dropout=0.0,
            max_seq_len=128
        ).cuda()

        x = torch.randint(0, 1000, (4, 64)).cuda()

        memories = []
        for num_iters in [2, 4, 6, 8]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            output, _ = model(x, force_iterations=num_iters)
            loss = output.sum()
            loss.backward()

            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memories.append((num_iters, peak_memory))
            model.zero_grad()

        # Check that memory grows roughly linearly
        # (Allow some variance, but should not be quadratic)
        if len(memories) >= 2:
            # Compute ratio of memory increase vs iteration increase
            mem_ratio = memories[-1][1] / memories[0][1]
            iter_ratio = memories[-1][0] / memories[0][0]

            # For O(N), mem_ratio should be close to iter_ratio
            # For O(N²), mem_ratio would be close to iter_ratio²
            # Allow up to 1.5x iter_ratio (some overhead is expected)
            assert mem_ratio < iter_ratio * 1.5, \
                f"Memory scaling looks worse than O(N): {mem_ratio:.2f}x memory for {iter_ratio:.1f}x iterations"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
