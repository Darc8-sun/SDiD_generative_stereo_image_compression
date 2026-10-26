"""Frozen OpenCLIP text encoder used as the cross-attention context.

Trimmed from the Latent-Diffusion / RDEIC encoder zoo: only the multi-embedder
text tower is kept, the image tower (and its kornia preprocessing) is dropped
because SDiD conditions on text only.
"""

import torch
import torch.nn as nn
import open_clip as clip


class FrozenOpenCLIPMultiEmbedder(nn.Module):
    """Returns the *token sequence* (B, 77, width) instead of the pooled vector.

    ``layer='penultimate'`` stops the text transformer one block early, which is
    what SDiD's cross-attention was trained with.
    """

    LAYERS = ["last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        max_length=77,
        freeze=True,
        layer="penultimate",
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version
        )
        self.model = model
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = 0 if layer == "last" else 1

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text)
        z = self.encode_with_transformer(
            tokens.to(next(self.model.parameters()).device)
        )
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return self.model.ln_final(x)

    def text_transformer_forward(self, x, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
