# SPDX-License-Identifier: Apache-2.0
# einsum tutorial: https://www.youtube.com/watch?v=pkVwUVEHmfI

import torch
from einops import rearrange


#data
BATCH_SIZE = 1
SEQ_LEN = 3
# Number of heads - each with Q,K,V.
num_heads = 1
head_dim = 4
qkv = 3


# Single tensor with all that.
init_tensor = torch.arange(1, BATCH_SIZE*SEQ_LEN*num_heads*qkv*head_dim+1)
print("init_tensor.shape :", init_tensor.shape)
print(init_tensor)
init_tensor_reshaped = rearrange(
    init_tensor,
    "(BATCH_SIZE num_heads qkv head_dim SEQ_LEN) -> qkv BATCH_SIZE num_heads head_dim SEQ_LEN",
    BATCH_SIZE=BATCH_SIZE, num_heads=num_heads, qkv=qkv, SEQ_LEN=SEQ_LEN
    )
print("init_tensor_reshaped.shape :", init_tensor_reshaped.shape)

queries, keys, values = init_tensor_reshaped[0], init_tensor_reshaped[1], init_tensor_reshaped[2]
print("queries.shape :", queries.shape)
print(queries)

# Q * K => score [BATCH_SIZE num_heads head_dim SEQ_LEN]
#score = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
#score = torch.einsum('bhds, bhds->bhsd', queries, keys)
#print("score.shape :", score.shape)
#print(score)

Q = rearrange(torch.arange(1,7), "(batch seq) -> batch seq", batch=BATCH_SIZE)
K = rearrange(torch.arange(1,7), "(batch seq) -> batch seq", batch=BATCH_SIZE)
V = rearrange(torch.arange(1,7), "(batch seq) -> batch seq", batch=BATCH_SIZE)
