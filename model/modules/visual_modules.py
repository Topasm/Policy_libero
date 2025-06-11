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
        self.vit.config.image_size = cfg.image_size

        # [MODIFIED] Use the num_latents value from the config
        self.resampler = PerceiverResampler(
            dim=self.vit.config.hidden_size,
            depth=2,
            num_latents=cfg.num_latents_per_image,  # 설정값 사용
            heads=8,
            dim_head=64,
        )
        self.projection = nn.Linear(
            self.vit.config.hidden_size, cfg.image_latent_dim)

    def forward(self, image_tensor_3ch):
        resized_images = TF.resize(
            image_tensor_3ch, [self.cfg.image_size, self.cfg.image_size], antialias=True)

        outputs = self.vit(pixel_values=resized_images)
        image_feats = outputs.last_hidden_state[:, 1:, :]  # Ignore CLS token

        resampled_feats = self.resampler(image_feats)
        projected_feats = self.projection(resampled_feats)

        # [MODIFIED] Return the whole sequence of latent vectors, instead of the mean.
        # Shape: (B, num_latents, image_latent_dim) e.g. (B, 64, 512)
        return projected_feats


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

    # [MODIFIED] Reverted to single-latent-input, single-image-output structure.
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """ Decodes a single latent vector into a single image. """
        x = self.initial_linear(latents).view(-1, 512, 4, 4)
        return self.decoder(x)
