# topasm/policy_libero/Topasm-Policy_libero-a2a8188ac53056b09728df3cc7753bfacd9df8c1/model/modules/visual_modules.py
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange, repeat

# [MODIFIED] Import from open_clip instead of transformers
from open_clip import create_model_from_pretrained, get_tokenizer

from config.config import VisionEncoderConfig, LanguageEncoderConfig
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, AutoImageProcessor


def exists(val):
    return val is not None


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def rearrange_many(args, pattern, **kwargs):
    return [rearrange(x, pattern, **kwargs) for x in args]


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale
        sim = torch.einsum("... i d, ... j d -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(self, *, dim, depth=3, dim_head=64, heads=8, num_latents=9, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PerceiverAttention(
                        dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ])
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # x is shape (B, N, D)
        b, n, d = x.shape
        latents = repeat(self.latents, "n d -> b n d", b=b)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class ImageEncoder(nn.Module):
    def __init__(self, cfg: VisionEncoderConfig):
        super().__init__()
        self.cfg = cfg
        # [MODIFIED] Load the processor associated with the vision backbone
        self.processor = AutoImageProcessor.from_pretrained(
            cfg.vision_backbone)
        self.vit = AutoModel.from_pretrained(cfg.vision_backbone)
        self.vit.config.image_size = cfg.image_size

        self.resampler = PerceiverResampler(
            dim=self.vit.config.hidden_size,
            depth=2, num_latents=cfg.num_query_per_image, heads=8, dim_head=64,
        )
        # projection layers removed

    def forward(self, image_tensor_3ch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [MODIFIED] Now returns raw features from ViT and Resampler without projection.
        """
        # The processor handles resizing and normalization (mean/std standardization).
        # It expects a list of images or a batched tensor.
        # It returns a dict with 'pixel_values'.
        inputs = self.processor(images=image_tensor_3ch, return_tensors="pt")

        # Move the processed tensor to the correct device
        processed_images = inputs['pixel_values'].to(image_tensor_3ch.device)

        # Pass the correctly preprocessed images to the ViT
        all_tokens = self.vit(pixel_values=processed_images).last_hidden_state

        cls_token = all_tokens[:, :1, :]
        patch_tokens = all_tokens[:, 1:, :]

        # Resample patch tokens but do not project them yet.
        resampled_patch_tokens = self.resampler(
            patch_tokens)  # Shape: (B, num_latents, 768)

        return resampled_patch_tokens, cls_token  # Return both types of tokens


# --- [MODIFIED] Rewritten LanguageEncoder using open_clip ---
class LanguageEncoder(nn.Module):
    def __init__(self, cfg: LanguageEncoderConfig):
        super().__init__()
        # Load model and tokenizer from open_clip
        self.model, _ = create_model_from_pretrained(
            model_name=cfg.model_name,
            pretrained=cfg.pretrained
        )
        self.tokenizer = get_tokenizer(cfg.model_name)

        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, texts):
        # Move model to the correct device
        device = next(self.model.parameters()).device

        text_tokens = self.tokenizer(texts).to(device)
        text_features = self.model.encode_text(text_tokens)

        # Return the raw features from the encoder
        return text_features.to(torch.float32)


class ImageDecoder(nn.Module):
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        self.initial_linear = nn.Sequential(
            nn.Linear(config.image_latent_dim, 512 * 4 * 4), nn.ReLU())

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2,
                               padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2,
                               padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2,
                               padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,
                               padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2,
                               padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, config.image_channels,
                               kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )

    # [MODIFIED] The forward method now accepts two latents and returns two images.
    def forward(self, latent_front: torch.Tensor, latent_wrist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Decodes two latent vectors into two separate images by reusing the same decoder weights. """
        # Decode front image
        x_front = self.initial_linear(latent_front).view(-1, 512, 4, 4)
        pred_front = self.decoder(x_front)

        # Decode wrist image
        x_wrist = self.initial_linear(latent_wrist).view(-1, 512, 4, 4)
        pred_wrist = self.decoder(x_wrist)

        return pred_front, pred_wrist
