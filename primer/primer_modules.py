# SPDX-License-Identifier: Apache-2.0
# Based on: https://arxiv.org/pdf/2109.08668.pdf
# Hints from: https://nn.labml.ai/transformers/primer_ez/index.html
# Shapes: https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853

import torch
from torch import Tensor, nn
import math
from einops import rearrange

from torchsummary import summary

from .transformer_modules import ResidualAdd

class SquaredReLUFeedForwardBlock(nn.Module):
    """Squared root ReLU.
    Changed the (default) emb_size to 384.
    Changed the (default) project expansion to 12.
    """
    def __init__(self, emb_size: int = 384, expansion: int = 12, drop_p: float = 0.0):

        super().__init__()

        self.upwards_proj = nn.Linear(emb_size, expansion * emb_size)
        self.act_fn = nn.ReLU()
        self.drop = nn.Dropout(drop_p)
        self.downwards_proj = nn.Linear(expansion * emb_size, emb_size)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        x = self.self.upwards_proj(x)
        # Squared ReLU
        x = self.act_fn(x)
        x = torch.square(x)
        # Dropout.
        x = self.drop(x)
        x = self.downwards_proj(x)
        return x

class MultiDConvHeadAttention(nn.Module):
    """ S1: Self-attention.
    
    Multi-depthwise-convolution-head self-attention module.

    Shrinked the (default) dimensions of embeddings to 384.
    Added depthwise convolutions.
    Scaling?
    """

    def __init__(self, emb_size: int = 384, num_heads: int = 12, dropout: float = 0.1):
        """ # Initializes MHA with default settings for ViT-Base. """
        super().__init__()
        # Model embeddings size (d_model).
        self.emb_size = emb_size
        # Number of heads.
        self.num_heads = num_heads

        # Check "head dim" = number of features per head (d_k).
        self.head_dim = emb_size // num_heads
        assert self.head_dim * num_heads == self.emb_size, "emb_size must be divisible by num_heads"
        # Calculate scaling factor.
        self.scale = 1 / math.sqrt(self.head_dim)

        # In vanilla self-attention Q,K,V are square matrices.
        # self.queries = nn.Linear(emb_size, emb_size)
        # self.keys = nn.Linear(emb_size, emb_size)
        # self.values = nn.Linear(emb_size, emb_size)
        self.qkv = nn.Linear(emb_size, emb_size * 3)

        # Depthwise convolutions.
        self.dconv_queries = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1, groups=self.head_dim)
        self.dconv_keys = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1, groups=self.head_dim)
        self.dconv_values = nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1, groups=self.head_dim)

        # Dropout and projection layer.
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)


    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        # Orig transformer.
        # qkv = rearrange(self.qkv(x), "batch_size seq_length (num_heads head_dim qkv) -> (qkv) batch_size num_heads seq_length head_dim", h=self.num_heads, qkv=3)
        # [ batch num_heads, seq_len]

        # Split keys, queries and values in [K|Q|V, batch, head_dim, num_heads, seq_length]
        qkv = rearrange(self.qkv(x), "batch_size seq_length (num_heads head_dim qkv) -> (qkv) batch_size head_dim num_heads seq_length", num_heads=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]


        import pdb;pdb.set_trace()
        # Apply spatial depthwise convolutions.
        queries = self.dconv_queries(queries)
        keys = self.dconv_keys(keys)
        values = self.dconv_values(values)

        ### S4: Attention Softmax.
        # Sum up over the last axis [batch, num_heads, query_len, key_len].
        score = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        # Apply mask - if provided.
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            score.mask_fill(~mask, fill_value)
        ### S4: end.

        # Scaled softmax.
        att = nn.functional.softmax(score * self.scale, dim=-1) 
        # Dropout.
        att = self.att_drop(att)

        # Sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Projection.
        out = self.projection(out)
        return out


class PrimerEncoderBlock(nn.Sequential):
    """ "S0: Main" - the block of Primer encoder.
    
     """
    def __init__(
        self,
        emb_size: int = 384,
        drop_p: float = 0.0,
        forward_expansion: int = 12,
        forward_drop_p: float = 0.0,
        **kwargs
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size), # "S5: Custom Norm" layer =  "S6: Normalization" + "S7: Scale-shift"
                    MultiDConvHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                    )
            ),
            ResidualAdd(
                nn.Sequential(
                    # (a) The normalization (S5) and feed forward (S2) subprograms switch places.
                    SquaredReLUFeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.LayerNorm(emb_size), # "S5: Custom Norm"
                    #nn.Dropout(drop_p),  # No dropout in Primer.
                )
            ),
        )


class PrimerEncoder(nn.Sequential):
    def __init__(self, layers: int = 12, **kwargs):
        super().__init__(*[PrimerEncoderBlock(**kwargs) for _ in range(layers)])



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
    encoder = PrimerEncoder(layers=NUM_LAYERS, emb_size=EMBEDDINGS_SIZE, num_heads=NUM_HEADS)
    summary(encoder, (SEQ_LEN, EMBEDDINGS_SIZE), device='cpu')

    embeddings = encoder(input)
    print("encoder embeddings:", embeddings.shape)    

