# topasm/fastpolicy/Topasm-FastPolicy-3497a74b08000a12da190edcbd9aa1864417e667/model/predictor/config.py
#!/usr/bin/env python3
"""
Unified configuration for the entire policy learning framework.
This class consolidates all parameters for different models (Hierarchical Transformer,
Inverse Dynamics, State Diffusion) and training settings into a single source of truth.
"""
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from lerobot.common.datasets.utils import PolicyFeature
from lerobot.configs.types import NormalizationMode, FeatureType


@dataclass
class VisionEncoderConfig:
    """Configuration for the vision encoder."""
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: Optional[str] = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    image_channels: int = 3
    image_size: int = 84  # Input size for the encoder
    output_image_size: int = 96  # Output size for the decoder
    crop_shape: Optional[Tuple[int, int]] = (84, 84)
    crop_is_random: bool = True
    image_latent_dim: int = 512  # Latent dimension for image features after projection


@dataclass
class HierarchicalTransformerConfig:
    """Configuration for the Hierarchical Autoregressive Transformer policy."""
    state_dim: int = 2  # Default for PushT, will be overridden
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    forward_steps: int = 32
    backward_steps: int = 32


@dataclass
class InverseDynamicsConfig:
    """Configuration for the Inverse Dynamics model."""
    hidden_dim: int = 512
    dropout: float = 0.1
    use_layernorm: bool = True
    out_activation: str = "Tanh"  # Will be converted to nn.Tanh()


@dataclass
class StateDiffusionConfig:
    """Configuration specific to the State-prediction Diffusion model."""
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    # Note: Transformer dims are inherited from HierarchicalTransformerConfig for consistency


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    dataset_repo_id: str = "lerobot/pusht"
    n_obs_steps: int = 3
    horizon: int = 16
    n_action_steps: int = 8  # How many actions to execute from a plan
    # What the diffusion model predicts: "action" or "observation.state"
    diffusion_target_key: str = "observation.state"
    interpolate_state: bool = True  # Critical for state prediction

    input_features: Dict[str, Any] = field(default_factory=dict)
    output_features: Dict[str, Any] = field(default_factory=dict)

    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )


@dataclass
class TrainingConfig:
    """Configuration for training."""
    training_steps: int = 5000
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    log_freq: int = 100
    save_freq: int = 1000
    num_workers: int = 4
    # For cosine scheduler
    lr_scheduler_T_max_mult: int = 1  # T_max = training_steps * mult


@dataclass
class PolicyConfig:
    """Unified configuration class for the entire project."""
    # Sub-configurations for modularity
    vision_encoder: VisionEncoderConfig = field(
        default_factory=VisionEncoderConfig)
    hierarchical_transformer: HierarchicalTransformerConfig = field(
        default_factory=HierarchicalTransformerConfig)
    inverse_dynamics: InverseDynamicsConfig = field(
        default_factory=InverseDynamicsConfig)
    state_diffusion: StateDiffusionConfig = field(
        default_factory=StateDiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self):
        """Recursively converts the config to a dictionary."""
        return asdict(self)

    def save_pretrained(self, output_dir: str | Path):
        """Save the configuration to a JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        # Convert enums to string values for JSON serialization
        def enum_to_str(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    enum_to_str(v)
                elif isinstance(v, NormalizationMode):
                    d[k] = v.value
            return d

        with open(output_dir / "config.json", "w") as f:
            json.dump(enum_to_str(config_dict), f, indent=2, sort_keys=True)
        print(f"Configuration saved to {output_dir / 'config.json'}")

    @classmethod
    def from_pretrained(cls, output_dir: str | Path):
        """Load the configuration from a JSON file."""
        config_path = Path(output_dir) / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Helper to reconstruct nested dataclasses from dict
        def from_dict_to_dataclass(data_class, data):
            # Convert string back to Enum
            if "normalization_mapping" in data:
                for k, v_str in data["normalization_mapping"].items():
                    data["normalization_mapping"][k] = NormalizationMode(
                        v_str)

            field_types = {
                f.name: f.type for f in data_class.__dataclass_fields__.values()}
            return data_class(**{
                f: from_dict_to_dataclass(field_types[f], data[f]) if hasattr(
                    field_types[f], '__dataclass_fields__') else data[f]
                for f in data
            })

        return from_dict_to_dataclass(cls, config_dict)
