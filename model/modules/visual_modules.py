# topasm/policy_libero/Topasm-Policy_libero-a2a8188ac53056b09728df3cc7753bfacd9df8c1/model/modules/visual_modules.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF  # ADDED: For image resizing
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
        full_seq = torch.cat([latents, x], dim=1)
        output = self.transformer_encoder(full_seq)
        return self.projection(output[:, :self.latents.shape[0], :])


class ImageEncoder(nn.Module):
    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.cfg = cfg  # Store config to access image_size in forward
        self.vit = AutoModel.from_pretrained(cfg.vision_backbone)

        patch_embeddings = self.vit.embeddings.patch_embeddings

        # --- FINAL FIX: Directly update the num_channels attribute on the submodule ---
        # This attribute is used for the validation check that was causing the error.
        patch_embeddings.num_channels = cfg.image_channels
        # --- END FIX ---

        original_conv = patch_embeddings.projection

        new_conv = nn.Conv2d(
            in_channels=cfg.image_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        with torch.no_grad():
            sum_of_weights = original_conv.weight.data.sum(dim=1, keepdim=True)
            avg_weights = sum_of_weights / 3.0
            new_conv.weight.data = avg_weights.repeat(
                1, cfg.image_channels, 1, 1)
            if original_conv.bias is not None:
                new_conv.bias.data = original_conv.bias.data

        patch_embeddings.projection = new_conv

        self.resampler = PerceiverResampler(
            input_dim=self.vit.config.hidden_size,
            output_dim=cfg.image_latent_dim,
            num_latents=cfg.perceiver['num_latents'],
            num_layers=cfg.perceiver['num_layers'],
            num_heads=cfg.perceiver['num_heads'],
        )

    def forward(self, images):
        # --- FIXED: Resize images to the model's expected input size ---
        resized_images = TF.resize(
            images, [self.cfg.image_size, self.cfg.image_size], antialias=True)
        outputs = self.vit(pixel_values=resized_images)
        # --- END FIX ---
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
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config

        # --- FIXED: Modified architecture to output 256x256 images ---
        # Start from a 4x4 spatial dimension instead of 3x3
        self.initial_linear = nn.Sequential(
            nn.Linear(config.image_latent_dim, 512 * 4 * 4), nn.ReLU())

        self.decoder = nn.Sequential(
            # Input: (B, 512, 4, 4)
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                512), nn.ReLU(True),       # -> (B, 512, 8, 8)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                256), nn.ReLU(True),       # -> (B, 256, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                128), nn.ReLU(True),       # -> (B, 128, 32, 32)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                64), nn.ReLU(True),        # -> (B, 64, 64, 64)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                32), nn.ReLU(True),         # -> (B, 32, 128, 128)
            nn.ConvTranspose2d(32, config.image_channels, kernel_size=4, stride=2,
                               padding=1), nn.Sigmoid()            # -> (B, C, 256, 256)
        )
        # --- END FIX ---

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # Reshape to the new starting size of 4x4
        x = self.initial_linear(latents).view(-1, 512, 4, 4)
        return self.decoder(x)
