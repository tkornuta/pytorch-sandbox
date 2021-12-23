# we'll use three operations
from einops import rearrange, reduce, repeat

import torch

# Number of heads - each with Q,K,V.
num_heads = 3
head_dim = 4
qkv = 3

tensors_list = []
# Create params for "heads" - each with QKV, each of with head_dim
for i in range(num_heads):
    tensors_list.append(torch.reshape(torch.tensor(range(1, qkv*head_dim+1)) + i *(qkv*head_dim), (qkv, head_dim)))

print("[NUM_HEADS, QKV, HEAD_DIM] :")
tensor = torch.stack(tensors_list)
print(tensor)
print(tensor.size())

# Single tensor with all that.
init_tensor = torch.tensor(range(1, num_heads*qkv*head_dim+1))
print(init_tensor)
init_tensor_reshaped = rearrange(init_tensor, "(num_heads qkv head_dim) -> num_heads qkv head_dim", num_heads=num_heads, qkv=qkv)
print(init_tensor_reshaped)
# Make sure tensors are equal.
assert torch.equal(tensor, init_tensor_reshaped)

print("first head -> Q")
print(tensor[0][0])
print("second head -> K")
print(tensor[1][1])

print("[QKV, (NUM_HEADS HEAD_DIM)] :")
qkv = rearrange(tensor, "num_heads qkv head_dim -> qkv (num_heads head_dim)")
print(qkv)