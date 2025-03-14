import torch
import torch.distributed as dist
from copy import deepcopy
from oslo.torch.distributed import ParallelContext, ParallelMode
from oslo.torch.nn import Linear2D
from _utils import split_2d, split_1d_twice, gather_2d, gather_1d_twice


parallel_context = ParallelContext.from_torch(
    data_parallel_size=1,
    pipeline_parallel_size=1,
    tensor_parallel_size=4,
    tensor_parallel_mode="2d",
)

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)
summa_dim = parallel_context.get_world_size(ParallelMode.TENSOR_2D_COL)
input_ = torch.randn((4, 4)).cuda()
target = torch.randn((4, 4)).cuda()
dist.broadcast(input_, src=0)
dist.broadcast(target, src=0)

linear = torch.nn.Linear(4, 4).cuda()
w = deepcopy(linear.weight.data)
b = deepcopy(linear.bias.data)

out = linear(input_)
optimizer = torch.optim.Adam(linear.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

if parallel_context.get_global_rank() == 0:
    print(f"original output: \n{out}\n")
    print(f"original updated weight: \n{linear.weight.data}\n")
    print(f"original updated bias: \n{linear.bias.data}\n")

# split input_ into 0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]
input_ = split_2d(parallel_context, input_, summa_dim, col_first=True)
# split target into 0:[0, 0], 1:[0, 1], 2:[1, 0], 3:[1, 1]
target = split_2d(parallel_context, target, summa_dim, col_first=True)
# split weight into 0:[0, 0], 1:[1, 0], 2:[0, 1], 3:[1, 1]
w = split_2d(parallel_context, w, summa_dim, col_first=False)
# split bias into 0:[0], 1:[2], 2:[1], 3:[3]
b = split_1d_twice(parallel_context, b, summa_dim, dim=0)

linear_2d = Linear2D(4, 4, parallel_context)
linear_2d.weight.data = w
linear_2d.bias.data = b

out = linear_2d(input_)
optimizer = torch.optim.Adam(linear_2d.parameters(), lr=1e-3)
logits = torch.nn.MSELoss()(out, target)
logits.backward()
optimizer.step()

out = gather_2d(parallel_context, out, summa_dim, col_first=False)
w = gather_2d(parallel_context, linear_2d.weight.data, summa_dim, col_first=True)
b = gather_1d_twice(parallel_context, linear_2d.bias.data, summa_dim, dim=0)

if parallel_context.get_global_rank() == 0:
    print(f"parallel output: \n{out}\n")
    print(f"parallel updated weight: \n{w}\n")
    print(f"original updated bias: \n{b}\n")
