import random

import numpy as np

import torch

from oslo.torch.nn.parallel.expert_parallel.layers import FP32LinearGate
from oslo.torch.nn.parallel.expert_parallel.layers import Top1Router, Top2Router
from oslo.torch.nn.parallel.expert_parallel.utils import UniformNoiseSampler, NormalNoiseSampler

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
#print(f'token_inp {token_inp}')

# 3. Create Gate
gate = FP32LinearGate(d_model, num_experts)
#print(f'gate.weight : {gate.weight}')
#print(f'gate.weight type : {gate.weight.dtype}')

#router2 = Top2Router()

# 4. Gate Forward
gate_out = gate(token_inp)
#print(f'gate_out : {gate_out}')

# 5. Create and Forward Top1 Router
select_policy = "random"
noisy_func = UniformNoiseSampler()
router1 = Top1Router(select_policy=select_policy, noisy_func=None, drop_tks=False)
router_res = router1(gate_out)


# 6. Create and Forward Top2 Router
select_policy = "random"
noisy_func = NormalNoiseSampler(num_experts=num_experts)
router2 = Top2Router(noisy_func=noisy_func, drop_tks=False)
router_res = router2(gate_out)

