import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, CLIPTextModel
from model.predictor.config import VisionEncoderConfig, LanguageEncoderConfig


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for compressing a sequence of features.
    """

    def __init__(self, input_dim, output_dim, num_latents, num_layers, num_heads):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)
        # Cross-attend from latents to the input sequence x
        full_seq = torch.cat([latents, x], dim=1)
        output = self.transformer_encoder(full_seq)
        return self.projection(output[:, :self.latents.shape[0], :])


class ImageEncoder(nn.Module):
    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.vit = AutoModel.from_pretrained(cfg.vision_backbone)
        self.resampler = PerceiverResampler(
            input_dim=self.vit.config.hidden_size,
            output_dim=cfg.image_latent_dim,
            num_latents=cfg.perceiver.num_latents,
            num_layers=cfg.perceiver.num_layers,
            num_heads=cfg.perceiver.num_heads,
        )

    def forward(self, images):
        outputs = self.vit(pixel_values=images)
        image_feats = outputs.last_hidden_state
        resampled_feats = self.resampler(image_feats)
        return resampled_feats.mean(dim=1)


class LanguageEncoder(nn.Module):
    def __init__(self, cfg: LanguageEncoderConfig):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(cfg.model_name)
        self.projection = nn.Linear(cfg.embedding_dim, cfg.projection_dim)

    def forward(self, texts):
        inputs = self.tokenizer(texts, padding=True,
                                return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_outputs = self.text_encoder(**inputs)
        pooled_output = text_outputs.pooler_output
        return self.projection(pooled_output)


class ImageDecoder(nn.Module):
    def __init__(self, config: HierarchicalPolicyConfig):
        super().__init__()
        self.config = config
        self.initial_linear = nn.Sequential(
            nn.Linear(config.image_latent_dim, 512 * 3 * 3), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(
                512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(
                256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(
                128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(
                64), nn.ReLU(True),
            nn.ConvTranspose2d(64, config.image_channels, 4, 2, 1), nn.Sigmoid())

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.initial_linear(latents).view(-1, 512, 3, 3)
        return self.decoder(x)
