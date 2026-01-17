"""
Configuration for Recursive Transformer experiments on language tasks.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json


# GPT-2 tokenizer vocab size
GPT2_VOCAB_SIZE = 50257


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = GPT2_VOCAB_SIZE  # GPT-2 tokenizer vocab
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_iterations: int = 8
    dropout: float = 0.1
    max_seq_len: int = 512


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_dir: str = "data/gsm8k"
    tokenizer_name: str = "gpt2"
    max_seq_len: int = 512
    batch_size: int = 16
    num_workers: int = 4
    simple_mode: bool = True  # Use GSM8KDatasetSimple


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    min_lr: float = 1e-6
    iteration_cost: float = 0.01
    done_supervision_weight: float = 0.5
    done_threshold: float = 0.7
    use_amp: bool = True
    gradient_clip: float = 1.0
    log_interval: int = 10
    save_dir: str = 'checkpoints'
    # Curriculum: list of (epoch_threshold, max_iters) tuples
    curriculum: Optional[List] = None


# Default curriculum: gradually unlock more iterations
DEFAULT_CURRICULUM = [(10, 2), (20, 4), (30, 6), (None, 8)]


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    name: str = 'recursive_transformer_gsm8k'
    seed: int = 42
    device: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to flat dictionary."""
        return {
            # Model
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'n_layers': self.model.n_layers,
            'd_ff': self.model.d_ff,
            'max_iterations': self.model.max_iterations,
            'dropout': self.model.dropout,
            'max_seq_len': self.model.max_seq_len,

            # Data
            'data_dir': self.data.data_dir,
            'tokenizer_name': self.data.tokenizer_name,
            'batch_size': self.data.batch_size,
            'num_workers': self.data.num_workers,
            'simple_mode': self.data.simple_mode,

            # Training
            'epochs': self.train.epochs,
            'learning_rate': self.train.learning_rate,
            'weight_decay': self.train.weight_decay,
            'min_lr': self.train.min_lr,
            'iteration_cost': self.train.iteration_cost,
            'done_supervision_weight': self.train.done_supervision_weight,
            'done_threshold': self.train.done_threshold,
            'use_amp': self.train.use_amp,
            'gradient_clip': self.train.gradient_clip,
            'log_interval': self.train.log_interval,
            'save_dir': self.train.save_dir,
            'curriculum': self.train.curriculum,

            # Meta
            'name': self.name,
            'seed': self.seed,
        }

    def save(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        config = cls()

        # Model
        config.model.vocab_size = data.get('vocab_size', config.model.vocab_size)
        config.model.d_model = data.get('d_model', config.model.d_model)
        config.model.n_heads = data.get('n_heads', config.model.n_heads)
        config.model.n_layers = data.get('n_layers', config.model.n_layers)
        config.model.d_ff = data.get('d_ff', config.model.d_ff)
        config.model.max_iterations = data.get('max_iterations', config.model.max_iterations)
        config.model.dropout = data.get('dropout', config.model.dropout)
        config.model.max_seq_len = data.get('max_seq_len', config.model.max_seq_len)

        # Data
        config.data.data_dir = data.get('data_dir', config.data.data_dir)
        config.data.tokenizer_name = data.get('tokenizer_name', config.data.tokenizer_name)
        config.data.batch_size = data.get('batch_size', config.data.batch_size)
        config.data.num_workers = data.get('num_workers', config.data.num_workers)
        config.data.simple_mode = data.get('simple_mode', config.data.simple_mode)

        # Training
        config.train.epochs = data.get('epochs', config.train.epochs)
        config.train.learning_rate = data.get('learning_rate', config.train.learning_rate)
        config.train.weight_decay = data.get('weight_decay', config.train.weight_decay)
        config.train.min_lr = data.get('min_lr', config.train.min_lr)
        config.train.iteration_cost = data.get('iteration_cost', config.train.iteration_cost)
        config.train.done_supervision_weight = data.get('done_supervision_weight', config.train.done_supervision_weight)
        config.train.done_threshold = data.get('done_threshold', config.train.done_threshold)
        config.train.use_amp = data.get('use_amp', config.train.use_amp)
        config.train.gradient_clip = data.get('gradient_clip', config.train.gradient_clip)
        config.train.log_interval = data.get('log_interval', config.train.log_interval)
        config.train.save_dir = data.get('save_dir', config.train.save_dir)
        config.train.curriculum = data.get('curriculum', config.train.curriculum)

        # Meta
        config.name = data.get('name', config.name)
        config.seed = data.get('seed', config.seed)

        return config


# =============================================================================
# Preset Configurations
# =============================================================================

# Tiny config for quick testing (~1M params)
TINY_CONFIG = ExperimentConfig(
    model=ModelConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        max_iterations=4,
        max_seq_len=256
    ),
    data=DataConfig(
        batch_size=32,
        max_seq_len=256
    ),
    train=TrainConfig(
        epochs=10,
        log_interval=1,
    ),
    name='tiny'
)

# Small config (~10M params)
SMALL_CONFIG = ExperimentConfig(
    model=ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_iterations=8,
        max_seq_len=512
    ),
    data=DataConfig(
        batch_size=16,
        max_seq_len=512
    ),
    train=TrainConfig(
        epochs=30,
        learning_rate=1.5e-4,
        log_interval=5,
    ),
    name='small'
)

# Medium config (~50M params)
MEDIUM_CONFIG = ExperimentConfig(
    model=ModelConfig(
        d_model=768,
        n_heads=12,
        n_layers=8,
        d_ff=3072,
        max_iterations=8,
        max_seq_len=512
    ),
    data=DataConfig(
        batch_size=8,
        max_seq_len=512
    ),
    train=TrainConfig(
        epochs=50,
        learning_rate=1e-4,
        log_interval=5,
    ),
    name='medium'
)

# Large config (~100M params)
LARGE_CONFIG = ExperimentConfig(
    model=ModelConfig(
        d_model=1024,
        n_heads=16,
        n_layers=12,
        d_ff=4096,
        max_iterations=8,
        max_seq_len=512
    ),
    data=DataConfig(
        batch_size=4,
        max_seq_len=512
    ),
    train=TrainConfig(
        epochs=100,
        learning_rate=5e-5,
        log_interval=5,
    ),
    name='large'
)


def get_config(name: str) -> ExperimentConfig:
    """Get preset configuration by name."""
    configs = {
        'tiny': TINY_CONFIG,
        'small': SMALL_CONFIG,
        'medium': MEDIUM_CONFIG,
        'large': LARGE_CONFIG,
    }

    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")

    return configs[name]


# =============================================================================
# Baseline Configurations (for comparison)
# =============================================================================

@dataclass
class BaselineModelConfig:
    """Baseline model architecture (no recursion)."""
    vocab_size: int = GPT2_VOCAB_SIZE
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    n_repeats: int = 3  # Weight-tied repetitions
    dropout: float = 0.1
    max_seq_len: int = 512


@dataclass
class BaselineConfig:
    """Baseline experiment configuration."""
    model: BaselineModelConfig = field(default_factory=BaselineModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    name: str = 'baseline'
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'n_layers': self.model.n_layers,
            'd_ff': self.model.d_ff,
            'n_repeats': self.model.n_repeats,
            'dropout': self.model.dropout,
            'max_seq_len': self.model.max_seq_len,

            'data_dir': self.data.data_dir,
            'tokenizer_name': self.data.tokenizer_name,
            'batch_size': self.data.batch_size,

            'epochs': self.train.epochs,
            'learning_rate': self.train.learning_rate,
            'weight_decay': self.train.weight_decay,
            'use_amp': self.train.use_amp,
            'save_dir': self.train.save_dir,

            'name': self.name,
            'seed': self.seed,
            'model_type': 'baseline',
        }


BASELINE_SMALL_CONFIG = BaselineConfig(
    model=BaselineModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        n_repeats=3
    ),
    data=DataConfig(batch_size=16),
    train=TrainConfig(
        epochs=30,
        save_dir='checkpoints_baseline'
    ),
    name='baseline_small'
)


def get_baseline_config(name: str) -> BaselineConfig:
    """Get baseline configuration."""
    configs = {
        'small': BASELINE_SMALL_CONFIG,
    }
    if name not in configs:
        raise ValueError(f"Unknown baseline config: {name}. Available: {list(configs.keys())}")
    return configs[name]


# =============================================================================
# Standard Transformer Config (no weight tying, no recursion)
# =============================================================================

@dataclass
class StandardModelConfig:
    """Standard transformer (no recursion, no weight tying)."""
    vocab_size: int = GPT2_VOCAB_SIZE
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512


@dataclass
class StandardConfig:
    """Standard transformer experiment config."""
    model: StandardModelConfig = field(default_factory=StandardModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    name: str = 'standard'
    seed: int = 42

    def to_dict(self) -> dict:
        return {
            'vocab_size': self.model.vocab_size,
            'd_model': self.model.d_model,
            'n_heads': self.model.n_heads,
            'n_layers': self.model.n_layers,
            'd_ff': self.model.d_ff,
            'dropout': self.model.dropout,
            'max_seq_len': self.model.max_seq_len,

            'data_dir': self.data.data_dir,
            'tokenizer_name': self.data.tokenizer_name,
            'batch_size': self.data.batch_size,

            'epochs': self.train.epochs,
            'learning_rate': self.train.learning_rate,
            'use_amp': self.train.use_amp,
            'save_dir': self.train.save_dir,

            'name': self.name,
            'seed': self.seed,
            'model_type': 'standard',
        }


STANDARD_CONFIG = StandardConfig(
    model=StandardModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=12,
        d_ff=2048
    ),
    data=DataConfig(batch_size=16),
    train=TrainConfig(
        epochs=30,
        save_dir='checkpoints_standard'
    ),
    name='standard'
)


def get_standard_config() -> StandardConfig:
    """Get standard transformer config."""
    return STANDARD_CONFIG
