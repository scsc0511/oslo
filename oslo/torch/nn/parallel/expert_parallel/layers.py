import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import ProcessGroup

import math
from typing import Callable, Optional

from ._context import ExpertParallelContextInfo
from ._ops import OSLO_EP_KERNEL_FLAG
from ._ops import AllToAll, EPDispatch, EPCombine
from .utils import get_current_device, cum_sum_d0_minus_one, _ForceFP32Parameter, auto_cast_softmax
from .utils import UniformNoiseSampler, NormalNoiseSampler


class Top1Router(nn.Module):
    # TODO : Add Doc String
    """

    """

    def __init__(self,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 select_policy: str = "first",
                 noisy_func: Callable = None,
                 drop_tks: bool = True):
        super().__init__()

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

        self.min_capacity = min_capacity
        self.select_policy = select_policy
        self.noisy_func = noisy_func

        self.drop_tks = drop_tks

        assert select_policy in {"first", "random"}
        if select_policy == "random":
            device = get_current_device()
            lower_bound = torch.tensor(0.0, device=device)
            upper_bound = torch.tensor(1.0, device=device)

            self.uniform = torch.distributions.uniform.Uniform(low=lower_bound, high=upper_bound).rsample

    def get_capacity(self, logits_shape):

        capacity_factor = self.capacity_factor_train if self.training else self.capcity_factor_eval
        capacity = math.floor(capacity_factor * logits_shape[-2] / logits_shape[-1])

        capacity += capacity % 2
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return capacity

    def forward(self,
                inputs: torch.Tensor,
                use_kernel: bool = False,
                ep_group: Optional[ProcessGroup] = None):

        if self.noisy_func is not None and self.training:
            inputs = self.noisy_func(inputs)

        logits = auto_cast_softmax(inputs, dim=-1)
        num_experts = logits.size(-1)
        capacity = self.get_capacity(logits.shape)

        top1_idx = torch.argmax(inputs, dim=-1)
        mask = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)

        if self.training:
            me = torch.mean(logits, dim=0)
            ce = torch.mean(mask.float(), dim=0)
            l_aux = num_experts * torch.sum(me * ce)

            # TODO : Think About the way to add Auxiliary Loss
            #        with the Expert Parallel
            # ep_context = ExpertParallelContext.from_torch()
            # ep_context.add_loss(l_aux)
        elif not self.drop_tks:
            max_num = torch.max(torch.sum(mask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=ep_group)
            capacity = max_num.item()
        else:
            pass

        if self.select_policy == "random":
            rand_mask = mask * self.uniform(mask.shape)
            _, dispatch_idx = torch.topk(rand_mask, k=capacity, dim=0)
            mask = mask * torch.zeros_like(mask).scatter_(0, dispatch_idx, 1)
            ranks = cum_sum_d0_minus_one(mask)
        elif self.select_policy == "first":
            ranks = cum_sum_d0_minus_one(mask)
            mask = mask * torch.lt(ranks, capacity)
        else:
            raise NotImplementedError("No support such select policy yet.")

        ranks = torch.sum(mask * ranks, dim=-1)

        if use_kernel:
            mask = torch.sum(mask, dim=-1)
            mask = torch.stack([mask], dim=0).to(torch.int32)
            dest_idx = torch.stack([top1_idx * capacity + ranks], dim=0).to(torch.int32)
            return logits, mask, dest_idx, num_experts * capacity
        else:
            ranks = F.one_hot(ranks, num_classes=capacity)
            weight = mask * logits.type_as(inputs)
            combine_weights = weight.unsqueeze(2) * ranks.unsqueeze(1)
            sec_mask = combine_weights.bool()
            return combine_weights, sec_mask


class Top2Router(nn.Module):
    # TODO : Add Doc String
    """

    """

    def __init__(self,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 noisy_func: Callable = None,
                 drop_tks: bool = True):
        super().__init__()

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

        self.min_capacity = min_capacity

        self.noisy_func = noisy_func
        self.drop_tks = drop_tks

    def get_capacity(self, logits_shape):

        capacity_factor = self.capacity_factor_train if self.training else self.capacity_factor_eval
        capacity = math.floor(capacity_factor * logits_shape[-2] / logits_shape[-1])
        capacity += capacity % 2
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return capacity

    def forward(self, inputs: torch.Tensor, use_kernel: bool = False,
                ep_group: Optional[ProcessGroup] = None):
        if self.noisy_func is not None and self.training:
            inputs = self.noisy_func(inputs)

        logits = auto_cast_softmax(inputs, dim=-1)
        num_experts = logits.size(-1)
        capacity = self.get_capacity(logits.shape)

        top1_idx = torch.argmax(logits, dim=-1)
        mask1 = F.one_hot(top1_idx, num_classes=num_experts).to(torch.int32)

        logits_except1 = logits.masked_fill(mask1.bool(), float("-inf"))
        top2_idx = torch.argmax(logits_except1, dim=-1)
        mask2 = F.one_hot(top2_idx, num_classes=num_experts).to(torch.int32)

        cmask = (mask1 + mask2)

        if self.training:
            me = torch.mean(logits, dim=0)
            ce = torch.mean(cmask.float(), dim=0)

            l_aux = num_experts * torch.sum(me * ce) / 2.0  # div 2 to normalize it to 1

            # TODO : Think About the way to add Auxiliary Loss
            #        with the Expert Parallel
            # ep_context = ExpertParallelContext.from_torch()
            # ep_context.add_loss(l_aux)
        elif not self.drop_tks:
            max_num = torch.max(torch.sum(cmask, dim=0))
            dist.all_reduce(max_num, op=dist.ReduceOp.MAX, group=ep_group)
            capacity = max_num.item()
        else:
            pass

        rank1 = cum_sum_d0_minus_one(mask1)
        rank2 = cum_sum_d0_minus_one(mask2)
        rank2 += torch.sum(mask1, dim=-2, keepdim=True)

        mask1 *= torch.lt(rank1, capacity)
        mask2 *= torch.lt(rank2, capacity)

        rank1 = torch.sum(mask1 * rank1, dim=-1)
        rank2 = torch.sum(mask2 * rank2, dim=-1)

        if use_kernel:
            mask1 = torch.sum(mask1, dim=-1)
            mask2 = torch.sum(mask2, dim=-1)

            mask = torch.stack([mask1, mask2], dim=0).to(torch.int32)
            dest_idx = torch.stack([top1_idx * capacity + rank1, top2_idx * capacity + rank2])

            return logits, mask, dest_idx, num_experts * capacity
        else:
            weight1 = mask1 * logits.type_as(inputs)
            weight2 = mask2 * logits.type_as(inputs)

            rank1_sc = F.one_hot(rank1, num_classes=capacity)
            rank2_sc = F.one_hot(rank2, num_classes=capacity)

            combine_weight1 = weight1.unsqueeze(2) * rank1_sc.unsqueeze(1)
            combine_weight2 = weight2.unsqueeze(2) * rank2_sc.unsqueeze(1)

            combine_weight = combine_weight1 + combine_weight2
            sec_mask = combine_weight.bool()

            return combine_weight, sec_mask


class FP32LinearGate(nn.Module):

    def __init__(self, d_model: int, num_experts: int, scale: float = 0.1):
        super().__init__()

        device = get_current_device()
        self.weight = _ForceFP32Parameter(torch.empty(num_experts, d_model, device=device))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(scale / d_model))

        return

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight)


class ExpertParallelFrontBlock(nn.Module):
    # TODO : Add Doc String
    """

    """

    def __init__(self, in_features: int,
                 out_features: int,
                 num_experts: int,
                 ep_info: ExpertParallelContextInfo,
                 combine_info: dict,
                 top_k: int,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0,
                 min_capacity: int = 4,
                 noisy_policy: Optional[str] = None,
                 drop_tks: bool = True,
                 use_residual: bool = False,
                 residual_instance: Optional[nn.Module] = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        # Initialize Gate
        self.gate = FP32LinearGate(in_features, num_experts)

        # Create Router
        noisy_func = None
        if noisy_policy is not None:
            if noisy_policy == 'Jitter':
                noisy_func = UniformNoiseSampler()
            elif noisy_policy == 'Gaussian':
                noisy_func = NormalNoiseSampler(num_experts)
            else:
                raise NotImplementedError("Unsupported input noisy policy")

        if top_k == 1:
            router_cls = Top1Router
        elif top_k == 2:
            router_cls = Top2Router
        else:
            raise NotImplementedError("top_k > 2 is not supported yet")

        self.router = router_cls(capacity_factor_train=capacity_factor_train,
                                 capacity_factor_eval=capacity_factor_eval,
                                 min_capacity=min_capacity,
                                 noisy_func=noisy_func,
                                 drop_tks=drop_tks)

        # Create Residual
        self.use_residual = use_residual
        if use_residual:
            assert residual_instance is not None, "If use residual connection, Residual Instance must be given"
            self.residual_module = residual_instance

            # TODO : Communicate with ZERO
            # with no_shard_zero_context() :
            self.residual_mix = nn.Linear(in_features, 2, device=get_current_device())
        else:
            self.residual_module = None
            self.residual_mix = None

        # Set Kernel Mode
        self.use_kernel = True if OSLO_EP_KERNEL_FLAG else False

        self.combine_info = combine_info
        self.ep_info = ep_info

        # TODO : Initialize Parameters
        self.weight = nn.Parameter(torch.empty(ep_info.num_local_experts, in_features, out_features).contiguous())
        self.bias = nn.Parameter(torch.empty(ep_info.num_local_experts, 1, out_features))

        # TODO : Add ep_info to Parameters of Experts

    def front_expert_process(self, expert_input: torch.Tensor):
        # |expert_input| = (ep_size, num_local_experts, capacity, in_features)

        h = expert_input.size(-1)

        expert_input = expert_input.transpose(0, 1)
        expert_shape = expert_input.shape
        # |expert_input| = input_shape = (num_local_experts, ep_size, capacity, in_features)

        expert_input = expert_input.reshape(self.ep_info.num_local_experts, -1, h)
        # |expert_input| = (num_local_experts, ep_size * capacity, in_features)

        front_expert_output = torch.baddbmm(self.bias, expert_input, self.weight)
        # |front_expert_output| = (num_local_experts, ep_size * capacity, out_features)

        return front_expert_output, expert_shape

    def front_a2a_process(self, dispatch_data: torch.Tensor):
        expert_input = AllToAll.apply(dispatch_data, self.ep_info.ep_group)
        a2a_shape = expert_input.shape
        # |expert_input| = (num_experts = ep_size * num_local_experts, capacity, in_features)

        expert_input = expert_input.reshape(self.ep_info.ep_size, self.num_local_experts, -1, self.in_features)
        # |expert_input| = (ep_size, num_local_experts, capacity, in_features)

        front_expert_output, expert_shape = self.front_expert_process(expert_input)
        # |front_expert_output| = (num_local_experts, ep_size * capacity, out_features)

        return front_expert_output, expert_shape, a2a_shape

    def forward(self, inputs: torch.Tensor):
        # |inputs| = (sentence_len, batch_size, in_features)

        tokens = inputs.reshape(-1, self.in_features)
        fp32_input = tokens.to(torch.float32) if inputs.dtype != torch.float32 else tokens
        # |tokens| = ( token_num, in_features )
        # |fp32_input| = ( token_num, in_features )

        gate_output = self.gate(fp32_input)
        # |gate_output| = ( token_num, in_features )

        # Routing Tokens
        router_res = self.router(inputs=gate_output, use_kernel=self.use_kernel, ep_group=self.ep_info.ep_group)

        # Dispatch
        if self.use_kernel:
            dispatch_data = EPDispatch.apply(tokens, *router_res[1:])
            dispatch_data = dispatch_data.reshape(self.num_experts, -1, self.in_features)
            # |dispatch_data| = (num_experts = , capacity, in_features)
        else:
            sec_mask_f = router_res[1].type_as(inputs)
            dispatch_data = torch.matmul(sec_mask_f.permute(1, 2, 0), tokens)
            # |dispatch_data| = (num_experts, capacity, in_features)

        front_output, expert_shape, a2a_shape = self.front_a2a_process(dispatch_data)
        # |front_output| = (num_local_experts, ep_size * capacity, out_features)

        # Residual Connection
        if self.use_residual:
            residual_inter = self.residual_module(inputs)
            # |residual_inter| = (sentence_len, batch_size, out_features)

            residual_weight = self.residual_mix(inputs)
            residual_weight = F.softmax(residual_weight, dim=-1)
            # |residual_weight| = (sentence_len, batch_size, 2)
        else:
            residual_output, residual_weight = None, None

        # Save Information for Combine
        self.combine_info['inputs_shape'] = inputs.shape

        self.combine_info['expert_shape'] = expert_shape
        self.combine_info['a2a_shape'] = a2a_shape

        self.combine_info['router_res'] = router_res

        self.combine_info['residual_inter'] = residual_inter
        self.combine_info['residual_weight'] = residual_weight

        return front_output


class ExpertParallelBehindBlock(nn.Module):
    # TODO : Add Doc String
    """

    """

    def __init__(self, in_features: int,
                 out_features: int,
                 num_experts: int,
                 ep_info: ExpertParallelContextInfo,
                 combine_info: dict,
                 use_residual: bool = False,
                 residual_instance: Optional[nn.Module] = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        # Initialize Module for Residual Connection
        self.use_residual = use_residual
        if use_residual:
            assert residual_instance is not None, "If use residual connection, Residual Instance must be given"
            self.residual_module = residual_instance
        else:
            self.residual_module = None
            self.residual_mix = None

        # Set Kernel Mode
        self.use_kernel = True if OSLO_EP_KERNEL_FLAG else False

        self.ep_info = ep_info
        self.combine_info = combine_info

        # Initialize Experts and Parallel Info
        self.weight = nn.Parameter(torch.empty(ep_info.num_local_experts, in_features, out_features).contiguous())
        self.bias = nn.Parameter(torch.empty(ep_info.num_local_experts, 1, out_features))

        # TODO : Add ep_info to Parameter of Experts

    def behind_expert_process(self, expert_input: torch.Tensor):
        # |expert_input| = (num_local_experts, ep_size * capacity, in_features)

        behind_expert_output = torch.baddbmm(self.bias, expert_input, self.weight)
        # |behind_expert_output| = (num_local_experts, ep_size * capacity, out_features)

        behind_expert_output = behind_expert_output.reshape(self.combine_info['expert_shape'])
        # |behind_expert_output| = (num_local_experts, ep_size, capacity, out_features)

        behind_expert_output = behind_expert_output.transpose(0, 1).contiugous()
        # |behind_expert_output| = (ep_size, num_local_experts, capacity, out_features)

        return behind_expert_output

    def behind_a2a_process(self, front_expert_output: torch.Tensor):
        # |front_expert_output| = (num_local_experts, ep_size * capacity, in_features)

        behind_expert_output = self.behind_expert_process(front_expert_output)
        # |behind_expert_output| = (ep_size, num_local_experts, capacity, out_features)

        behind_expert_output = behind_expert_output.reshape(self.combine_info['a2a_shape'])
        # |behind_expert_output| = (num_experts = ep_size * num_local_experts, capacity, out_features)

        behind_expert_output = AllToAll.apply(behind_expert_output, self.ep_info.ep_group)
        # |behind_expert_output| = (num_experts = ep_size * num_local_experts, capacity, out_features)

        return behind_expert_output

    def forward(self, inputs):
        # |inputs| = (sentence_len, batch_size, in_features)

        behind_expert_output = self.behind_a2a_process(inputs)
        # |behind_expert_output| = (num_experts = ep_size * num_local_experts, capacity, out_features)

        # Combine
        if self.use_kernel:
            behind_expert_output = behind_expert_output.reshape(-1, self.out_features)
            # |expert_output| = (num_experts * capacity, out_features)

            output = EPCombine.apply(behind_expert_output, *self.combine_info['router_res'])
            # |output| = (num_experts * capacity, out_features)
        else:
            combine_weights = self.combine_info['router_res'][0].type_as(inputs)
            combine_weights = combine_weights.view(combine_weights.shape[0], -1)
            behind_expert_output = behind_expert_output.view(-1, behind_expert_output.shape(-1))
            output = torch.matmul(combine_weights, behind_expert_output)
        output = output.reshape(self.combine_info['inputs_shape'])
        # |output| = ( sentence_len, batch_size, in_features )

        if self.use_residual:
            residual_inter = self.combine_info['residual_inter']
            residual_output = self.residual_module(residual_inter)
            output = output * self.combine_info['residual_weight'][..., 0:1] \
                     + residual_output * self.combine_info['residual_weight'][..., 1:]

        return output
