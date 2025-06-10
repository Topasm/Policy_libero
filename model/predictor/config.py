# topasm/policy_libero/Topasm-Policy_libero-a2a8188ac53056b09728df3cc7753bfacd9df8c1/model/predictor/config.py
#!/usr/bin/env python3
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from lerobot.common.datasets.utils import PolicyFeature
from lerobot.configs.types import NormalizationMode, FeatureType


@dataclass
class VisionEncoderConfig:
    """Configuration for the vision encoder."""
    vision_backbone: str = "google/vit-base-patch16-224"
    image_size: int = 224  # ADDED: To specify the input size for the ViT model
    image_latent_dim: int = 512
    image_channels: int = 6
    perceiver: Dict[str, Any] = field(default_factory=lambda: {
        "num_latents": 64,
        "num_layers": 2,
        "num_heads": 8
    })


@dataclass
class LanguageEncoderConfig:
    """Configuration for the language encoder (e.g., CLIP)."""
    model_name: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    projection_dim: int = 512


@dataclass
class HierarchicalTransformerConfig:
    """Configuration for the Hierarchical Autoregressive Transformer policy."""
    state_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    forward_steps: int = 32
    backward_steps: int = 32
    num_goal_tokens: int = 1
    num_bwd_tokens: int = 1
    num_fwd_tokens: int = 1


@dataclass
class InverseDynamicsConfig:
    """Configuration for the Inverse Dynamics model."""
    hidden_dim: int = 512
    dropout: float = 0.1
    use_layernorm: bool = True
    out_activation: str = "Tanh"


@dataclass
class StateDiffusionConfig:
    """Configuration specific to the State-prediction Diffusion model."""
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    dataset_repo_id: str = "yongjincho/libero"
    # ADDED: Define the image keys to be loaded from the dataset
    image_keys: List[str] = field(default_factory=lambda: [
                                  "observation.images.front", "observation.images.wrist"])
    n_obs_steps: int = 3
    horizon: int = 16
    n_action_steps: int = 8
    diffusion_target_key: str = "observation.state"
    interpolate_state: bool = True
    state_delta_indices: List[int] = field(
        default_factory=lambda: list(range(-1, 16)))
    action_delta_indices: List[int] = field(
        default_factory=lambda: list(range(16)))
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
    training_steps: int = 20000
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    log_freq: int = 100
    save_freq: int = 1000
    num_workers: int = 4
    lr_scheduler_T_max_mult: int = 1


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    project: str = "policy-libero-project"
    entity: Optional[str] = None
    group: str = "bidirectional-transformer"
    mode: str = "online"


@dataclass
class PolicyConfig:
    """Unified configuration class for the entire project."""
    wandb: WandbConfig = field(default_factory=WandbConfig)
    vision_encoder: VisionEncoderConfig = field(
        default_factory=VisionEncoderConfig)
    language_encoder: LanguageEncoderConfig = field(
        default_factory=LanguageEncoderConfig)
    hierarchical_transformer: HierarchicalTransformerConfig = field(
        default_factory=HierarchicalTransformerConfig)
    inverse_dynamics: InverseDynamicsConfig = field(
        default_factory=InverseDynamicsConfig)
    state_diffusion: StateDiffusionConfig = field(
        default_factory=StateDiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self):
        # Convert dataclasses to nested dictionaries
        d = asdict(self)
        # Manually handle serialization of PolicyFeature objects
        for features_key in ["input_features", "output_features"]:
            if features_key in d["data"]:
                d["data"][features_key] = {
                    k: {"type": v.type.name, "shape": v.shape}
                    for k, v in d["data"][features_key].items()
                }
        return d

    def save_pretrained(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_dict = self.to_dict()

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
        config_path = Path(output_dir) / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # [FIXED] Manually reconstruct PolicyFeature objects from dictionaries
        if "data" in config_dict:
            for features_key in ["input_features", "output_features"]:
                if features_key in config_dict["data"]:
                    reconstructed_features = {}
                    for key, feature_dict in config_dict["data"][features_key].items():
                        if isinstance(feature_dict, dict) and "type" in feature_dict:
                            # Convert string like "STATE" back to Enum FeatureType.STATE
                            feature_type_enum = FeatureType[feature_dict["type"]]
                            reconstructed_features[key] = PolicyFeature(
                                type=feature_type_enum,
                                shape=tuple(feature_dict["shape"])
                            )
                        else:
                            reconstructed_features[key] = feature_dict
                    config_dict["data"][features_key] = reconstructed_features
        # --- END FIX ---

        def from_dict_to_dataclass(data_class, data):
            if "normalization_mapping" in data:
                for k, v_str in data["normalization_mapping"].items():
                    data["normalization_mapping"][k] = NormalizationMode(v_str)

            field_types = {
                f.name: f.type for f in data_class.__dataclass_fields__.values()}

            # Filter out keys that are not in the dataclass definition
            valid_keys = {f for f in field_types if f in data}

            return data_class(**{
                f: from_dict_to_dataclass(field_types[f], data[f])
                if hasattr(field_types[f], '__dataclass_fields__') and isinstance(data.get(f), dict)
                else data.get(f)
                for f in valid_keys
            })

        return from_dict_to_dataclass(cls, config_dict)
