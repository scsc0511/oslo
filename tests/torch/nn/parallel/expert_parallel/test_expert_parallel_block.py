import random

import numpy as np

import torch

from oslo.torch.nn.parallel.expert_parallel.layers import ExpertParallelFrontBlock, ExpertParallelBehindBlock

torch.set_printoptions(threshold=10_000)

token_num = 32
d_model = 2
num_experts = 4


# 1. Set Random Seed for Reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 2. Generate Input
token_inp = torch.rand(token_num, d_model).to(torch.float32)
