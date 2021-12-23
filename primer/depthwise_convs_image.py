# SPDX-License-Identifier: Apache-2.0
# Based on: https://www.paepper.com/blog/posts/depthwise-separable-convolutions-in-pytorch/


import torch
from torch.nn import Conv2d

conv = Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1)
params = sum(p.numel() for p in conv.parameters() if p.requires_grad)

x = torch.rand(5, 10, 50, 50)
print("input size: ", x.size())

out = conv(x)
print("conv output size: ", out.size())

depth_conv = Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1, groups=10)
point_conv = Conv2d(in_channels=10, out_channels=32, kernel_size=1)

depthwise_separable_conv = torch.nn.Sequential(depth_conv, point_conv)
params_depthwise = sum(p.numel() for p in depthwise_separable_conv.parameters() if p.requires_grad)

out_depthwise = depthwise_separable_conv(x)
print("depthwise_conv (only) output size: ", depth_conv(x).size())
print("depthwise_separable_conv output size: ", out_depthwise.size())

print(f"The standard convolution uses {params} parameters.")
print(f"The depthwise separable convolution uses {params_depthwise} parameters.")

assert out.shape == out_depthwise.shape, "Size mismatch"