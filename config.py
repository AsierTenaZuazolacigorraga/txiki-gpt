import math
import os
import pickle
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

dataset = "shakespeare_char"
init_from = "scratch"

# seed
seed = 1337
torch.manual_seed(seed)

# logs
eval_interval = 2000
eval_iters = 20
log_interval = 1

# data
gradient_accumulation_steps = 1  # simulate larger batch size
batch_size = 12
block_size = 64
vocab_size = 50304

# model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False

# training
learning_rate = 1e-3
max_iters = 2000

# optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# learning rate decay
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 1e-4
decay_lr = True

# system
pt_dtype = torch.float16
device = "cpu"
device_type = "cpu"

# sample
start = "\n"
num_samples = 10
max_new_tokens = 500
temperature = (
    0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)

# I/O
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", dataset)
os.makedirs(out_dir, exist_ok=True)
ckpt_path = os.path.join(out_dir, "ckpt.pt")
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", dataset)
os.makedirs(data_dir, exist_ok=True)
meta_path = os.path.join(data_dir, "meta.pkl")

# overrides
if dataset == "shakespeare_char":
    vocab_size = 65

elif dataset == "shakespeare":
    if init_from == "gpt2":

        # logs
        eval_interval = 5
        eval_iters = 40

        # data
        batch_size = 1
        gradient_accumulation_steps = 32

        # training
        max_iters = 20

        # learning rate decay
        learning_rate = 3e-5
        decay_lr = False
