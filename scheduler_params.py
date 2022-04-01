import torch
from torch.optim import Adam
from nemo.core.optim.lr_scheduler import NoamAnnealing
import numpy as np
import matplotlib.pyplot as plt

params = torch.zeros(1, 768, requires_grad=True)

# TRAINING Info
num_epochs = 100
batch_size = 10

initial_lr = 5.0  # (LR scaler for Noam)
warmup_ratio = 0.1
d_model = params.size(-1)  # change this to play around with peak

# NODE Info
num_gpus = 1
num_nodes = 1

# DATASET Info
num_files = 10

"""
COMPUTATION
"""
steps = int(num_files * num_epochs / num_gpus / num_nodes / batch_size)
warmup_steps = steps * warmup_ratio

optim = Adam([params], lr=initial_lr)
policy = NoamAnnealing(optim, max_steps=steps, warmup_ratio=warmup_ratio, d_model=d_model)

x = [i for i in range(steps)]
y = []
for step in x:
    y.append(policy.get_lr())
    policy.step()

y = np.asarray(y)
print("Num steps :", steps)
print("Peak step :", y.argmax(axis=0))
print("Peak LR :", y.max(axis=0))
print("Final LR :", y[-1])

plt.plot(x, y, label='Policy={}'.format(str(policy.__class__.__name__)))
plt.title(f"Noam schedule (lr scaler={initial_lr}, warmup={warmup_steps}, d_model={d_model})")
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.legend(loc='upper right')
plt.show()
