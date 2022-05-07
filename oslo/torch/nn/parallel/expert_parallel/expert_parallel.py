import copy

import torch
import torch.distributed as dist
import torch.nn as nn

from oslo.torch.distributed import ParallelContext, ParallelMode, seed
from oslo.torch.nn.parallel.expert_parallel._context import ExpertParallelContext, ExpertParallelContextInfo
from oslo.torch.nn.parallel.expert_parallel.mapping import ExpertParallelMapping

from oslo.torch.nn.parallel.utils import is_huggingface_model
from oslo.torch.nn.parallel.utils import ParallelWrapper
from oslo.transformers.mapping_utils import _ExpertParallelMappingForHuggingFace

from oslo.torch.nn.parallel.expert_parallel._ops import OSLO_EP_KERNEL_FLAG

from oslo.torch.nn.parallel.expert_parallel.layers import Top1Router, Top2Router, FP32LinearGate
from oslo.torch.nn.parallel.expert_parallel.layers import ExpertParallelFrontBlock
from oslo.torch.nn.parallel.expert_parallel.layers import ExpertParallelBehindBlock

from oslo.torch.nn.parallel.expert_parallel.utils import UniformNoiseSampler, NormalNoiseSampler


def _update_module_arguments(module, **kwargs):
    for k, v in kwargs.items():
        setattr(module, k, v)


class ExpertParallel(ParallelWrapper):
    def __init__(
            self,
            model: nn.Module,
            parallel_context: ParallelContext,
            max_ep_size=dist.get_world_size(),
            num_experts: int = 0,
            top_k: int = 2,
            capacity_factor_train: float = 1.25,
            capacity_factor_eval: float = 2.0,
            min_capacity: int = 4,
            noisy_policy: str = None,
            drop_tks: bool = True,
            use_residual: bool = None
    ):
        super().__init__()
        self.model = model
        self.ep_context = ExpertParallelContext(parallel_context, max_ep_size).setup(seed(ParallelMode.TENSOR))
        self.device = torch.cuda.current_device()

        self.num_experts = num_experts
        if num_experts < 1:
            self.num_experts = parallel_context.get_world_size()

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval
        self.min_capacity = min_capacity
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_policy = noisy_policy
        self.drop_tks = drop_tks

        if noisy_policy is None:
            noisy_policy = 'Jitter' if use_residual else 'Gaussian'
        self.noisy_policy = noisy_policy

        if use_residual is None:
            use_residual = True if top_k == 1 else False
        self.use_residual = use_residual

        if is_huggingface_model(model):
            mapping = _ExpertParallelMappingForHuggingFace().get_mapping(model)
        else:
            raise ValueError(
                "`mapping` must be input if the model is not huggingface model."
            )

        self.use_kernel = True if OSLO_EP_KERNEL_FLAG else False

        self.expert_parallel_mapping = ExpertParallelMapping(mapping)
        self._parallelize()

    @torch.no_grad()
    def _parallelize(self):
        self._parallelize_module()

        # TODO : Need to Change _postproces()
        self._postprocess()
        _update_module_arguments(self.module, parallel_context=self.parallel_context)

    def _extract_combine_info_key(self, module_name):
        spl_modules = module_name.split('.')

        split_id = len(spl_modules)
        for i, cur_module in enumerate(spl_modules):
            if cur_module.isdigit():
                split_id = i + 1

        return '.'.join(spl_modules[:split_id])

    def _create_router(self, capacity_factor_train, capacity_factor_eval, min_capacity,
                       num_experts, top_k, noisy_policy, drop_tks):

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

        return router_cls(capacity_factor_train=capacity_factor_train,
                          capacity_factor_eval=capacity_factor_eval,
                          min_capacity=min_capacity,
                          noisy_func=noisy_func,
                          drop_tks=drop_tks)


    def _wrap_front(self, module: nn.Module, module_name: str):
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_1D)

        out_features, in_features = module.weight.size()

        gate = FP32LinearGate(in_features, self.num_experts)
        router = self._create_router(self.capacity_factor_train, self.capacity_factor_eval, self.min_capacity,
                                     self.num_experts, self.top_k, self.noisy_policy, self.drop_tks)
        if self.use_residual:
            # TODO : Add Activation in the Front Block or Behind Block
            residual_module = copy.deepcopy(module)
            residual_mix = nn.Linear(in_features, 2)
        else:
            residual_module, residual_mix = None, None

        # Add Cur Module's Combine Info
        combine_info_k = self._extract_combine_info_key(module_name)
        if combine_info_k not in self.combine_info:
            self.combine_info[combine_info_k] = dict()

        ep_group = self.parallel_context.get_group(ParallelMode.TENSOR_1D)
        ep_info = ExpertParallelContextInfo(num_experts=self.num_experts, ep_group=ep_group)

        _update_module_arguments(
            module=module,
            parallel_context=self.parallel_context,
            in_features=in_features, out_features=out_features,
            num_experts=self.num_experts,
            # TODO: add seacrh for expert_parallel only module for from_pretrained
            expert_parallel_gate=gate, expert_parallel_router=router,
            residual_use=self.use_residual,
            expert_parallel_residual=residual_module, expert_parallel_residual_mix=residual_mix,
            use_kernel=self.use_kernel,
            combine_info=self.combine_info[combine_info_k],
            ep_info=ep_info)

        if hasattr(module, "weight") and module.weight is not None:
            new_param = nn.Parameter(torch.empty(ep_info.num_local_experts, in_features, out_features).contiguous())
            param_list = new_param.chunk(ep_info.num_local_experts, dim=0)
            module.weight = param_list[rank]

        if hasattr(module, "bias") and module.bias is not None:
            new_param = nn.Parameter(torch.empty(ep_info.num_local_experts, 1, out_features).contiguous())
            param_list = new_param.chunk(ep_info.num_local_experts, dim=0)
            module.bias = param_list[rank]

        return module

    def _wrap_behind(self, module, module_name):
        rank = self.parallel_context.get_local_rank(ParallelMode.TENSOR_1D)

        out_features, in_features = module.weight.size()

        if self.use_residual:
            residual_module = copy.deepcopy(module)
        else:
            residual_module, residual_mix = None, None

        # Add Cur Module's Combine Info
        combine_info_k = self._extract_combine_info_key(module_name)
        if combine_info_k not in self.combine_info:
            self.combine_info[combine_info_k] = dict()
        ep_group = self.parallel_context.get_group(ParallelMode.TENSOR_1D)
        ep_info = ExpertParallelContextInfo(num_experts=self.num_experts, ep_group=ep_group)

        _update_module_arguments(
            module=module,
            parallel_context=self.parallel_context,
            in_features=in_features, out_features=out_features,
            num_experts=self.num_experts,
            residual_use=self.use_residual, residual_module=residual_module,
            use_kernel=self.use_kernel,
            ep_info=ep_info,
            combine_info=self.combine_info[combine_info_k])

        if hasattr(module, "weight") and module.weight is not None:
            new_param = nn.Parameter(torch.empty(self.num_experts, in_features, out_features).contiguous())
            param_list = new_param.chunk(ep_info.num_local_experts, dim=0)
            module.weight = param_list[rank]

        if hasattr(module, "bias") and module.bias is not None:
            new_param = nn.Parameter(torch.empty(self.num_experts, 1, out_features))
            param_list = new_param.chunk(ep_info.num_local_experts, dim=0)
            module.bias = param_list[rank]

        return module

    def _parallelize_module(self):
        for module_name, module in self.module.named_modules():
            if self.expert_parallel_mapping.is_front_parallel(self.module, module_name):
                self._wrap_front(module, module_name)
                module.__class__ = ExpertParallelFrontBlock
            elif self.expert_parallel_mapping.is_behind_parallel(self.module, module_name):
                self._wrap_behind(module, module_name)
                module.__class__ = ExpertParallelBehindBlock

    # TODO : Need to Replace
    def _postprocess(self):
        for param in self.parameters():
            if not param.is_cuda and torch.is_tensor(param):
                param.data = param.to(self.device)

        for param in self.buffers():
            if not param.is_cuda and torch.is_tensor(param):
                param.data = param.to(self.device)


