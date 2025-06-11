# topasm/policy_libero/Topasm-Policy_libero-a2a8188ac53056b09728df3cc7753bfacd9df8c1/model/modules/visual_modules.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPTextModel
from einops import rearrange, repeat
# [MODIFIED] Import for resizing
import torchvision.transforms.functional as TF
from model.predictor.config import VisionEncoderConfig, LanguageEncoderConfig


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
    def __init__(self, *, dim, depth=2, dim_head=64, heads=8, num_latents=64, ff_mult=4):
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
        self.vit = AutoModel.from_pretrained(cfg.vision_backbone)

        # [MODIFIED] The ViT now takes 3-channel images, so no need to adapt the first layer.
        # We only need to ensure its config is correct for resizing if needed.
        self.vit.config.image_size = cfg.image_size

        # [MODIFIED] Use the new, more sophisticated PerceiverResampler
        self.resampler = PerceiverResampler(
            dim=self.vit.config.hidden_size,
            # These parameters can be tuned in the future
            depth=2,
            num_latents=64,
            heads=8,
            dim_head=64,
        )

        # A projection layer to get the final embedding size
        self.projection = nn.Linear(
            self.vit.config.hidden_size, cfg.image_latent_dim)

    def forward(self, image_tensor_3ch):  # Expects (B, 3, H, W)
        # [FIXED] Resize input images to the size expected by the ViT model.
        resized_images = TF.resize(
            image_tensor_3ch, [self.cfg.image_size, self.cfg.image_size], antialias=True)

        outputs = self.vit(pixel_values=resized_images)

        # ViT output shape: (B, num_patches+1, hidden_size). We ignore the CLS token.
        image_feats = outputs.last_hidden_state[:, 1:, :]

        # Resample visual tokens into a fixed number of latents
        # (B, num_latents, hidden_size)
        resampled_feats = self.resampler(image_feats)

        # Project the latents to the desired dimension and take the mean
        # (B, num_latents, image_latent_dim)
        projected_feats = self.projection(resampled_feats)
        return projected_feats.mean(dim=1)  # (B, image_latent_dim)


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
