import torch

# values from:
# https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a
# https://github.com/The-AI-Summer/self-attention-cv/blob/main/self_attention_cv/transformer_vanilla/self_attention.py

# SEQ_LEN = 5, embeddings_size = 4
input = torch.tensor([[1, 0, 1, 0],
    [0, 2, 0, 2],
    [0, 2, 0, 2],
    [0, 2, 0, 2],
    [1, 1, 1, 1]], dtype=torch.float64)

# Construct "batch" [ BATCH=1, SEQ_LEN=2, embeddings_size=4 ]
#input = torch.stack([input, input], dim=0)
input = torch.unsqueeze(input, dim=0)
print("input.shape: ", input.shape)
print(input)

# Simulate "head QKV" projection layer.
Wq = torch.tensor(
[[1, 0, 1],
 [1, 0, 0],
 [0, 0, 1],
 [0, 1, 1]], dtype=torch.float64)

Wk = torch.tensor(
[[0, 0, 1],
 [1, 1, 0],
 [0, 1, 0],
 [1, 1, 0]], dtype=torch.float64)

Wv = torch.tensor(
[[0, 2, 0],
 [0, 3, 0],
 [1, 0, 3],
 [1, 1, 0]], dtype=torch.float64)

import pdb;pdb.set_trace()

# Calculate QKV for a given batch.
queries = torch.matmul(input, Wq)
print("queries.shape: ", queries.shape)
print(queries)

keys = torch.matmul(input, Wk)
values = torch.matmul(input, Wv)

# Dot product attention: [batch, seq_len, seq_len]
scores =  torch.matmul(queries, keys.transpose(-2,-1))
print("score.shape: ", scores.shape)
print(scores)

# Calculate attention.
att = torch.nn.functional.softmax(scores, dim=-1)

# Output values.
out = torch.matmul(att, values)
print("out shape: ", out.shape)
print(out)

# Version 2: with torch.einsum
# (Not scaled) Dot attention - resulting shape: [batch, seq_len, seq_len]
dot_prod = torch.einsum('b i d , b j d -> b i j', queries, keys) # no scalling here.
attention = torch.softmax(dot_prod, dim=-1)
# output - resulting shape [batch, seq_len, emb_size]
out_ein = torch.einsum('b i j , b j d -> b i d', attention, values)

print("out_ein shape: ", out_ein.shape)
print(out_ein)

assert torch.equal(out, out_ein)
print("results equal!")