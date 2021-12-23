# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn import Conv1d, Conv2d, Conv3d

seq_len = 197
batch_size = 2
num_heads = 12
d_k = 64
# embeddings_size = num_heads * d_k # 768
# alias:
head_dim = d_k

# https://arxiv.org/pdf/2109.08668.pdf
# The convolution is along the sequence dimension and per channel (depth-wise).
# Apply D−Conv to head: x = d_conv(x, width=3, head_size=hs, axis="spatial", mask="causal")
# Spatial D-Conv 3x1
# CONV 3X1 tf.nn.conv1d input - - filters
# DCONV 3X1 tf.nn.depthwise_conv2d input - - filters !!!!
# DEPTHWISE_CONV_3X1 In0: 17 In1: 10 Dim: 384 C: -1.12

# CONV 1X1 tf.layers.dense inputs - - units !!! this is just a dense layer!


# input from projection layer [seq_len, batch_size, heads, d_k]
# my: [batch, head_dim, num_heads, seq_length]
# https://nn.labml.ai/transformers/primer_ez/index.html
# labml.ai: [batch_size, heads, d_k, seq_len]
query = torch.rand(batch_size, num_heads, head_dim, seq_len)
print("query size: ", query.size())

# Depthwise convolution: 
# * groups=in_channels,
# * each input channel is convolved with its own set of filters (of size in_channels/ out_channels = 1).
# * padding = 1 -> dims remains the same.
depth_conv = Conv1d(in_channels=num_heads, out_channels=num_heads, kernel_size=(1,3), padding=(0,1), groups=num_heads)
print(depth_conv)

params_depthwise = sum(p.numel() for p in depth_conv.parameters() if p.requires_grad)
print(f"The depthwise separable convolution uses {params_depthwise} parameters.")


print("depthwise_conv (only) output size: ", depth_conv(query).size())

# https://nn.labml.ai/transformers/primer_ez/index.html
# seq_len, batch_size, heads, d_k = x.shape
# x = x.permute(1, 2, 3, 0)
# batch_size, heads, d_k, seq_len