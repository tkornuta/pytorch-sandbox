# SPDX-License-Identifier: Apache-2.0

import torch
from einops import repeat, rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor, nn

from torchsummary import summary


class PositionalEmbedding(nn.Module):
    """Module reponsible for adding (hardcoded) positional embeddings to the input sequence."""
    def position_encoding(self, seq_len: int, embeddings_size: int) -> Tensor:
        pos = torch.arange(seq_len, dtype=torch.float).reshape(1, -1, 1)
        dim = torch.arange(embeddings_size, dtype=torch.float).reshape(1, 1, -1)
        phase = pos / 1e4 ** (dim // embeddings_size)
        return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

    def forward(self, x: Tensor) -> Tensor:
        seq_len, dimension = x.size(1), x.size(2)
        x += self.position_encoding(seq_len, dimension)
        return x

class PatchEmbedding(nn.Module):
    def __init__(
        self, img_size: int = 224, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768,
    ):
        super().__init__()

        # Create the projection layer.
        # self.projection = nn.Sequential(
        #    # Split image into s1 x s2 patches.
        #    Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
        #    nn.Linear(patch_size * patch_size * in_channels, emb_size)
        # )
        # Projection layer v2: convolutions for speed.
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        # A trainable [class] token "whose state at the output of the Transformer encoder serves as the image representation y".
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # A trainable 1D positional embeddings ([class] token + num patches)
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        # Get batch_size.
        batch_size = x.shape[0]

        # Create patch embeddings.
        patch_embeddings = self.projection(x)

        # Extend embeddings with class token added at front.
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=batch_size)
        cls_patch_embaddings = torch.cat([cls_tokens, patch_embeddings], dim=1)

        # Add positional embeddings.
        cls_patch_embaddings += self.positions

        return cls_patch_embaddings


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, emb_size: int = 768, num_heads: int = 12, dropout: float = 0.1):
        """ # Initializes MHA with default settings for ViT-Base. """
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        # Check "head dim" = number of features per head (d_k).
        self.head_dim = emb_size // num_heads
        assert self.head_dim * num_heads == self.emb_size, "emb_size must be divisible by num_heads"
        # Calculate scaling factor.
        self.scale = 1 / (self.head_dim ** 0.5)

        # V1: in vanilla self-attention Q,K,V are square matrices.
        # self.keys = nn.Linear(emb_size, emb_size)
        # self.queries = nn.Linear(emb_size, emb_size)
        # self.values = nn.Linear(emb_size, emb_size)
        
        # V2: single layer with emb_size, split into num (heads * head_dim) * 3 (Q,K,V).
        self.qkv = nn.Linear(emb_size, emb_size * 3)

        # Attention dropout.
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Split keys, queries and values into [batch, num_heads, seq_length, embedding_size]
        # queries = rearrange(self.queries(x), "b n (h e) -> b h n e", h=self.num_heads)
        # keys = rearrange(self.keys(x), "b n (h e) -> b h n e", h=self.num_heads)
        # values  = rearrange(self.values(x), "b n (h e) -> b h n e", h=self.num_heads)
        
        #import pdb;pdb.set_trace()
        # Split keys, queries and values in [K|Q|V, batch, head_dim, num_heads, seq_length]
        #qkv = rearrange(self.qkv(x), "batch_size seq_length (num_heads head_dim qkv) -> (qkv) batch_size head_dim num_heads seq_length", num_heads=self.num_heads, qkv=3)
        #queries, keys, values = qkv[0], qkv[1], qkv[2]


        qkv = rearrange(self.qkv(x), "b n (h e qkv) -> (qkv) b h n e", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        #import pdb;pdb.set_trace()
        # Sum up over the last axis [batch, num_heads, query_len, key_len].
        score = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        # Apply mask - if provided.
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            score.mask_fill(~mask, fill_value)

        # Scaled softmax [batch, num_heads, seq_len, seq_len]
        att = nn.functional.softmax(score * self.scale, dim=-1)
        # Dropout.
        att = self.att_drop(att)

        # Head concatenation - sum up over the third axis (embeddings dimension).
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Projection.
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.0):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size: int = 768,
        drop_p: float = 0.0,
        forward_expansion: int = 4,
        forward_drop_p: float = 0.0,
        **kwargs
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(nn.LayerNorm(emb_size), MultiHeadAttention(emb_size, **kwargs), nn.Dropout(drop_p))
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, layers: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(layers)])

    #def forward(self, x: Tensor) -> Tensor:
    #    import pdb;pdb.set_trace()
    #    return super().forward(x)


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'), nn.LayerNorm(emb_size), nn.Linear(emb_size, n_classes)
        )

# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
# https://arxiv.org/pdf/2010.11929.pdf
# Table 1: Table 1: Details of Vision Transformer model variants, page 5:
# Model     Layers  Hidden size D   MLP size    Heads   Params
# ViT-Base  12      768             3072        12      86M
# ViT-Large 24      1024            4096        16      307M
# ViT-Huge  32      1280            5120        16      632M

# Attention is all you need
# https://arxiv.org/pdf/1706.03762.pdf
# Table 3: Variations on the Transformer architecture, page  9:
# Model     N   d_model     d_ff    h   d_k     d_v     Params
# base      6   512         2048    8   64      64      65M

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
# https://arxiv.org/pdf/1810.04805.pdf
# BERT Base:  12 layers, 12 heads, 768 embeddings_size,  110M params
# BERT large: 24 layers, 16 heads, 1024 embeddings_size, 345M params

if __name__ == "__main__":

    # data dims.
    BATCH_SIZE = 51
    SEQ_LEN = 123
    # model params.
    EMBEDDINGS_SIZE = 768
    NUM_LAYERS = 12
    NUM_HEADS = 12

    input = torch.rand(BATCH_SIZE, SEQ_LEN, EMBEDDINGS_SIZE)
    #tgt = torch.rand(64, 16, 512)
    print("inputs:", input.shape)    

    # Test encoder.
    encoder = TransformerEncoder(layers=NUM_LAYERS, emb_size=EMBEDDINGS_SIZE, num_heads=NUM_HEADS)
    summary(encoder, (SEQ_LEN, EMBEDDINGS_SIZE), device='cpu')

    embeddings = encoder(input)
    print("encoder embeddings:", embeddings.shape)    

    # Test head.
    #head = ClassificationHead(emb_size=EMBEDDINGS_SIZE)
    #summary(head, (SEQ_LEN, EMBEDDINGS_SIZE), device='cpu')
    #output = head(embeddings)
    #print("head output:", output.shape)    
